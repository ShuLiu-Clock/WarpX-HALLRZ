# HallRZ AMR 网格与边界设计建议

日期：2026-06-08

本文从 `HallRZ_EB_ROBIN_REFACTOR_PLAN_CN.md` 和 Stage 3/4 测试中提炼出面向使用者的
网格设计建议。它不是求解器实现计划，而是用于 Python/benchmark/生产算例选网格、选 patch、
选边界条件和解释速度/误差 tradeoff 的设计手册。

## 0. 冻结版使用结论

当前冻结建议：

- 小网格优先单层：约 `224x256` 到 `448x512`，AMR 固定开销大于 cell 数收益。
- 中等网格优先两层：约 `672x768` 到 `896x1024`，两层最小开销低，实际 ratio 约 `3.6` 时已有收益。
- 大网格优先三层：约 `1120x1280` 起，三层开始稳定超过两层和单层；`2240x2560` 测得三层
  speedup 约 `3.76x`。
- 三层 target actual ratio 建议先取 `5-7`，当前数据中 ratio 约 `6` 是比较稳健的折中。
- 正式 PIC 不要把 refined patch 边界贴着 `hib/out`。level 1/2 边界应明显离开 channel wall、
  exit step、EB Neumann wall 和主要 sheath/强梯度区。
- 默认使用上一 PIC 步 fine `phi` 热启动：

```text
warpx.hall_rz_amr_phi_init_weight = 1.0
```

边界组合建议：

- EB：Neumann 作为压力测试和生产优先路径。
- physical r-hi/z-hi：Robin。
- z-lo 阳极/入口：Dirichlet。
- r-lo：`rmin=0` 时为 RZ axis；`rmin>0` 时按物理模型选择。

推荐验收指标：

- `phi/Er/Ez/rho` 无 NaN/Inf。
- `rho` 不被 Poisson solve 修改。
- `coarse_fine_phi_continuity.max_abs <= 1e-10`。
- 误差统计必须排除 HallRZ 非计算区域。
- 同时记录 `phi L2 rel`、`E L2 rel max`、`E Linf abs/rel`、最大误差位置和 patch margin。

## 1. 当前求解框架

当前 WarpX HallRZ AMR Poisson 路径是 level-by-level solve：

- 单层：只解 full finest 单层。
- 两层：先解 level 0，再把当前 coarse phi 插值到 level 1，作为 level 1 人工边界和初始猜测。
- 三层：依次解 level 0、level 1、level 2。
- fine level 的人工 high-r/high-z 边界必须使用当前 coarse 解的 Dirichlet 插值。
- fine patch 内部初始猜测使用

```text
phi_init = w * phi_fine_previous + (1-w) * Interp(phi_coarse_current)
```

其中 `warpx.hall_rz_amr_phi_init_weight = w`，默认 `w=1`。人工 Dirichlet 边界在混合后仍强制
刷新为当前 coarse 插值，不能被上一 PIC 步 fine phi 污染。

重要限制：

- 当前不是 AMReX composite AMR MLMG solve。
- 每个 AMR level 都会独立构造 linop/MLMG 并独立求解。
- 因此 AMR 是否更快，首先取决于 solve-cell ratio 是否足够大；如果 patch 太大，多层固定开销会
压过 cell 数减少。

## 2. 边界条件建议

推荐用于综合压力测试的边界组合：

- EB 边界：Neumann。EB Neumann 对收敛更有挑战性，适合做设计 gate。
- r-hi / z-hi 物理边界：Robin，AMReX raw contract 为 `a*phi + b*dphi/dcoord = f`。
- z-lo 阳极/入口：Dirichlet。
- r-lo：
  - `rmin = 0` 时是 RZ axis regularity/homogeneous Neumann，不是 EB，也不是 Robin。
  - `rmin > 0` 时可以按物理需要使用 Dirichlet/Neumann/Robin。

方向约定必须保持：

- physical domain face Robin/Neumann 使用 AMReX nodal solver 的 positive coordinate derivative，
  不是 outward-normal derivative。
- EB-FVM Neumann/Robin 使用 EB normal：fluid/computational region -> covered region。
- Python 端传入数据必须已经满足 AMReX raw contract；WarpX 不自动猜测外法向口径。

当前 Python 边界数据接口状态：

- EB Neumann / EB Robin 支持 AMR 分层传入：
  - `hallrz.set_eb_neumann(g, lev=lev)`
  - `hallrz.set_eb_robin(a, b, f, lev=lev)`
- 若 AMR 使用 Python EB 数据，必须为每个 active AMR level 都提供一份数据。两层需要 `lev=0,1`；
  三层需要 `lev=0,1,2`。
- 每层 EB 数组必须对应该层的 cell domain，shape 为 `(Nr_lev, Nz_lev)`，顺序为 WarpX RZ
  `(r,z)`。当前实现要求该层 cell domain zero-based。
- WarpX 不把 coarse EB 数据插值到 fine EB 数据，也不把 fine EB 数据限制到 coarse。Python 端必须按
  每层网格和每层 EB cut-cell 几何分别给出物理数据。
- 当前 AMR path 不支持 Python physical-domain boundary arrays
  (`set_robin_zhi/rhi/rlo`、`set_dirichlet_rlo`、`set_inlet_dirichlet`)。这些接口仍是单层数据；
  AMR 路径若检测到它们 active 会 abort。
- AMR physical-domain BC 目前使用 scalar/default 参数，在每一层 `ComputePhiAndE()` 内按该层
  nodal grid 和该层 geometry 生成。它不是从 coarse level physical BC 插值得到的。
- 只有 AMR artificial boundary 使用当前 coarse phi 的 Dirichlet 插值。不要把 artificial boundary
  和 physical domain boundary 混为一类。

未来若增加 Python physical-domain arrays 的 AMR 支持，应采用 per-level raw data 设计：每层数据
直接对应此层实际 physical boundary face nodes，而不是由 coarse physical BC 插值生成。只接触真实
物理 domain 的 face 使用 physical BC；fine patch 在物理域内部产生的 high-r/high-z 边界仍使用
current coarse phi Dirichlet。

## 3. solve-cell ratio 的含义

本文使用：

```text
full/AMR solve-cell ratio =
    full-finest cell count / sum(level-by-level AMR solve cell counts)
```

它是 cell 数减少给 AMR 带来的乐观速度空间，不包含：

- 多次独立 MLMG setup/prepare/solve 固定开销；
- coarse-to-fine phi 插值；
- AMR field/particle ownership 和通信；
- EB geometry / mask / flux record 构造；
- GPU kernel launch overhead。

因此：

- ratio 接近 1 时，AMR 基本不可能更快。
- ratio 只有 1.3-2 时，多层 AMR 往往仍慢。
- ratio 约 5-7 时，三层 AMR 才有现实机会超过 full-finest。
- 两层 AMR 理论上限小于 4，因为至少要完整解一次 level 0：

```text
two-level ratio = 1 / (1/4 + area_fraction_level1)
```

所以不要要求两层 AMR 达到 `actual ratio=5-7`；这是结构上不可达的。

三层 AMR 的近似模型为：

```text
three-level ratio =
    1 / (1/16 + area_fraction_level1/4 + area_fraction_level2)
```

三层只有在 level 1/2 patch 明显小于全域时才有速度收益。

## 4. patch 设计建议

基本几何约束：

- patch 必须 lower-left 对齐：`[rmin, r_i] x [zmin, z_i]`。
- refined patch 必须覆盖整个 thruster channel 和 exit step。
- 两层：

```text
hib < r1 < rmax
out < z1 < zmax
```

- 三层：

```text
hib < r2 < r1 < rmax
out < z2 < z1 < zmax
```

对齐约束：

- 两层：`r1/z1` 在 level-0 cell face；`lob/hib/out` 在 level-1 cell face。
- 三层：`r1/z1` 在 level-0 cell face；`r2/z2` 在 level-1 cell face；`lob/hib/out` 在 level-2
  cell face。

设计建议：

- 先根据 channel 尺寸确定最小可行 patch，再计算 actual ratio。
- 不要把 Ω1/Ω2 做成“几乎全域”；这会把 AMR 变成多次重复 solve。
- 如果目标是速度收益，优先扩大 plume/full domain 或减小 refined near-channel region，而不是在
  C++ 端强行限制 patch。
- patch 过小会提高 coarse-fine 人工边界误差；需要同时看速度和误差。

## 5. 当前已知测试结果

### 5.1 两层 AMR 小网格对照

测试口径：

- GPU, 200 steps
- full-finest reference: `140 x 160`
- two-level AMR: level 0 `70 x 80`, level 1 patch `r1=z1=0.06`
- EB Robin 历史测试口径，static RHS
- 误差只统计 AMR composite owned 区域和 HallRZ 计算区，不包含非计算区域。

结果：

```text
case              avg s/step     steady s/step
full 140x160      0.00336536     0.00326395
2-level AMR       0.00605903     0.00595687
```

```text
full/AMR solve-cell ratio = 1.087
steady speedup full/AMR   = 0.548
```

Masked error vs full-finest：

```text
field   L2 rel          Linf rel
phi     2.5957e-4       6.4343e-4
Er      1.1103e-2       9.5046e-2
Ez      1.2116e-3       2.5617e-2
```

判断：

- 正确性通过：finite、coarse-fine phi continuity、masked error 都正常。
- 速度不通过收益预期：ratio 只有 1.087，cell 数几乎没有减少，但多解了一个 level。
- 该例说明小网格/大 patch 下单层最快。

### 5.2 三层 AMR 2048-class 历史结果

已有 Stage 4.5/4.6 结果显示：

- `2048 x 2048` 方形 full-finest，在 deepest MG 下三层 AMR ratio 约 5 时仍未超过 full-finest。
- `2048 x 3072` 扩展域测试中，ratio 增大后 AMR 性能更接近 full-finest，但需要继续系统比较
  单层/两层/三层。
- `2048 x 4096` full-finest reference 在当前 8GB GPU 上出现 OOM，因此后续大网格测试需要记录
  memory gate；不能把 OOM 的 full reference 当作有效速度对照。

### 5.3 Stage 4.8：单层/两层/三层转变点

测试口径：

- GPU, 200 steps, static rho。
- 物理域：`rmax=0.28`，`zmax=0.32`。
- 边界：EB Neumann，r-hi/z-hi scalar Robin，z-lo scalar Dirichlet。
- deepest MG：单层和 AMR 都使用显式大 MCL upper bound，让 AMReX 自己选择实际 MG 深度。
- 误差：相对同一 full-finest reference；只统计 HallRZ computational region 和 AMR composite
  owned region，不统计非计算区域。
- 三层 target ratio 约 6；两层使用最小 channel-covering patch，因此 actual ratio 接近两层结构上限
  `3.6`。

速度和误差结果：

```text
full grid   fastest  one s/step  two speedup  three speedup  two E L2 rel  three E L2 rel  three phi L2 rel
224x256     one       0.00550159  0.667        0.492          2.459e-2      1.352e-2        1.088e-3
448x512     one       0.00787893  0.817        0.721          1.622e-2      6.888e-3        5.435e-4
672x768     two       0.0136807   1.378        1.077          1.234e-2      4.614e-3        3.404e-4
896x1024    two       0.0207752   1.646        1.558          1.013e-2      3.472e-3        2.420e-4
1120x1280   three     0.0287081   1.841        2.299          8.678e-3      2.782e-3        1.838e-4
1344x1536   three     0.0384021   2.004        2.438          7.650e-3      2.320e-3        1.465e-4
1568x1792   three     0.0424438   2.381        2.800          6.876e-3      1.990e-3        1.210e-4
1792x2048   three     0.0570733   2.376        2.604          6.270e-3      1.743e-3        1.022e-4
2240x2560   three     0.0879795   2.827        3.757          5.376e-3      1.395e-3        7.663e-5
```

读表结论：

- 小网格：`224x256`、`448x512` 下单层最快。固定开销和多次独立 solve 抵消了 AMR cell 数减少。
- 中等网格：`672x768`、`896x1024` 下两层最快。两层开销较低，ratio 约 `3.6` 已足够收益。
- 大网格：从 `1120x1280` 起三层最快。`2240x2560` 时三层 steady speedup 达到 `3.76x`。
- 三层的 `phi L2 rel` 通常比两层大，但 `E L2 rel max` 明显小于两层。这说明只看 phi 不能完整
  判断 HallRZ AMR 质量；生产设计应同时看 `phi`、`Er`、`Ez`。

最大误差位置：

- 两层最大 E 误差集中在 `Er`、level 1、`z=out=0.04` 附近，`r` 接近 high channel wall
  `hib=0.05`。例如 `2240x2560` 为 `0.847866 @ Er_fp/L1/r=0.0504375/z=0.04`。
- 三层最大 E 误差也在 `Er`、level 1、`z=out=0.04` 附近，但 `r` 位于 level 1 patch / coarse-fine
  人工边界相关位置。ratio 约 6 时 `2240x2560` 为
  `0.364366 @ Er_fp/L1/r=0.104375/z=0.04`。
- 当前最大误差位置都在 HallRZ computational region 内，不是非计算区域统计污染。

这里的 `0.847866` 是最大绝对误差，不是相对误差。对 `2240x2560` 两层 case：

```text
Er Linf abs error              = 0.847866
Er Linf rel to global |Er|max  = 3.935e-2
max-point reference Er         = -10.3840
max-point AMR Er               = -9.5361
max-point local relative error = 8.17e-2
E L2 rel max                   = 5.376e-3
```

这个误差高度局部：

```text
2240x2560 two-level Er abs-error distribution:
p99.99  = 5.09e-2
p99.999 = 1.53e-1
max     = 8.48e-1
abs(error) > 0.5: 1 point, about 7.7e-7 of counted Er samples
```

产生该 spike 的原因不是统计口径，而是测试中两层 patch 被推到几乎最小 channel-covering patch：

```text
two-level patch: r1=0.05025, z1=0.04025
channel wall/exit: hib=0.05, out=0.04
max Er error point: r=0.0504375, z=0.04
```

也就是说，fine patch 人工边界几乎贴着 high channel wall 和 exit step。该位置同时靠近 EB
Neumann、几何角点和强梯度区域，`phi` 的小误差经 `E=-grad(phi)` 放大成局部 E spike。正式 PIC
网格设计不应采用这种极限 patch；level 1/2 边界应明显离开 `hib/out`，让通道、EB Neumann wall、
出口台阶和主要 sheath/强梯度区域留在 refined patch 内部。具体 margin 不在 C++ 端硬编码，应由
物理尺度和收敛测试确定；测试报告至少需要同时记录 patch margin、最大误差位置和
`E Linf abs/rel`。

### 5.4 Stage 4.8：三层 target ratio 5/6/7 影响

代表性 200-step 结果：

```text
grid        target ratio  actual ratio  three speedup  phi L2 rel  E L2 rel max  E Linf abs location
896x1024    5             4.896         1.260          2.308e-4    3.410e-3      Er L1 r=0.120938 z=0.04
896x1024    6             5.853         1.558          2.420e-4    3.472e-3      Er L1 r=0.105938 z=0.04
896x1024    7             6.850         1.178          2.493e-4    3.579e-3      Er L1 r=0.092188 z=0.04
1344x1536   5             4.925         2.120          1.396e-4    2.278e-3      Er L1 r=0.119792 z=0.04
1344x1536   6             5.890         2.438          1.465e-4    2.320e-3      Er L1 r=0.104792 z=0.04
1344x1536   7             6.895         2.309          1.513e-4    2.392e-3      Er L1 r=0.092292 z=0.04
2240x2560   5             4.951         3.235          7.342e-5    1.370e-3      Er L1 r=0.119875 z=0.04
2240x2560   6             5.943         3.757          7.663e-5    1.395e-3      Er L1 r=0.104375 z=0.04
2240x2560   7             6.928         3.715          7.946e-5    1.437e-3      Er L1 r=0.091875 z=0.04
```

判断：

- `896x1024` 仍应优先两层；三层 target ratio 5/6/7 都没有超过两层。
- `1344x1536` 和 `2240x2560` 已进入三层收益区间，target ratio 5/6/7 都能超过两层或接近最优。
- 在当前测试中，ratio 约 6 是比较稳健的折中：速度接近最好，E 误差未明显恶化。
- ratio 越大通常 patch 越小，最大误差位置会沿 `z=out` 移动到更靠近中心的 coarse-fine/patch
  人工边界处；这应作为 patch 设计审查点。

## 6. 推荐测试矩阵

为了找出单层、两层、三层的转变点，建议使用同一物理域和同一 finest spacing，对比：

```text
1-level: full finest
2-level: L0 = full/2, one refined patch covering channel
3-level: L0 = full/4, L1/L2 lower-left patches, target actual ratio ~= 5-7
```

推荐扫描：

```text
full grid:
224 x 256
448 x 512
672 x 768
896 x 1024
1120 x 1280
1344 x 1536
1568 x 1792
1792 x 2048
2240 x 2560

domain:
rmax = 0.28
zmax = 0.32
```

说明：

- 推荐使用 `224*s x 256*s` 这一族网格，并保持 `rmax=0.28`、`zmax=0.32`。这能让
  `lob=0.035`、`hib=0.050`、`out=0.040` 在 finest 和 AMR solve 所需 coarse levels 上保持一致。
- 不要随意换成未检查全层对齐的 `512x1024`、`1024x2048`、`336x384` 等网格。`336x384` 的
  三层测试已经触发 coarse level 对齐失败，可作为负例。
- `2048 x 4096` 可作为 memory-limit probe，但当前 GPU 上 full reference 可能 OOM。
- 两层 AMR 使用最小 channel-covering patch 或接近其最大 feasible ratio 的 patch。
- 三层 AMR target ratio 取 `5`、`6`、`7`。
- 默认 `warpx.hall_rz_amr_phi_init_weight=1`。
- 误差统计必须使用 computational-region mask。
- 每个 case 输出最大误差位置：field、level、`r,z`、abs error、relative error。

## 7. 设计判据

建议冻结/推荐某个 AMR layout 前至少满足：

- 所有 level 的 `phi/E` finite，无 NaN/Inf。
- `coarse_fine_phi_continuity.max_abs <= 1e-10`。
- masked `phi L2 rel` 和 `E L2 rel max` 在可接受范围内。
- 最大误差位置可解释，不集中在非计算区域或未保护的 covered edge。
- 速度至少超过 full-finest reference；若目标是生产收益，建议 steady speedup >= 1.5。
- 若三层在 ratio 5-7 仍不能超过 full-finest，需要继续定位 independent level-by-level solve 的固定开销。

## 8. 使用方法与参数模板

### 8.1 单层

单层适合小网格、debug、full-finest reference 和误差对照。

```text
warpx.hall_rz_enable = 1
warpx.hall_rz_amr_enable = 0
warpx.hall_rz_max_coarsening_level = 30
```

单层使用 scalar `warpx.hall_rz_max_coarsening_level`。如果要比较不同 MG 深度，扫这个参数即可。

### 8.2 两层静态 rectangular AMR

两层适合中等网格。参数结构：

```text
amr.max_level = 1
amr.ref_ratio = 2

warpx.hall_rz_enable = 1
warpx.hall_rz_amr_enable = 1
warpx.hall_rz_amr_r1 = <r1>
warpx.hall_rz_amr_z1 = <z1>
warpx.hall_rz_max_coarsening_levels = <mcl0> <mcl1>
warpx.hall_rz_amr_phi_init_weight = 1.0
```

几何约束：

```text
hib < r1 < rmax
out < z1 < zmax
r1/z1 lie on level-0 cell faces
lob/hib/out lie on level-1 cell faces
```

设计建议：

- 不要使用只刚好覆盖 channel 的极限 patch，例如 `r1` 只比 `hib` 大一个 cell、`z1` 只比 `out`
  大一个 cell。这会把 AMR artificial Dirichlet 边界放到 EB Neumann 和强梯度区附近，容易产生局部
  `Er` spike。
- 若两层用于生产，先用扩大 patch margin 的算例重新与 full-finest 比较，不要直接采用本文件中
  “最小 channel-covering patch”的误差作为生产结论。

### 8.3 三层静态 rectangular AMR

三层适合大网格。参数结构：

```text
amr.max_level = 2
amr.ref_ratio = 2 2

warpx.hall_rz_enable = 1
warpx.hall_rz_amr_enable = 1
warpx.hall_rz_amr_r1 = <r1>
warpx.hall_rz_amr_z1 = <z1>
warpx.hall_rz_amr_r2 = <r2>
warpx.hall_rz_amr_z2 = <z2>
warpx.hall_rz_max_coarsening_levels = <mcl0> <mcl1> <mcl2>
warpx.hall_rz_amr_phi_init_weight = 1.0
```

几何约束：

```text
hib < r2 < r1 < rmax
out < z2 < z1 < zmax
r1/z1 lie on level-0 cell faces
r2/z2 lie on level-1 cell faces
lob/hib/out lie on level-2 cell faces
```

当前推荐从 target actual ratio 约 `6` 开始设计，再扫 `5` 和 `7`。如果最大 E 误差落在
coarse-fine 人工边界且不可接受，应优先扩大 patch margin，而不是先改 solver。

### 8.4 Python EB 数据

AMR 下 Python EB Neumann / Robin 数据必须按 level 传入。示意：

```python
from pywarpx import hallrz

hallrz.set_eb_neumann(g0, lev=0)
hallrz.set_eb_neumann(g1, lev=1)
hallrz.set_eb_neumann(g2, lev=2)
```

或 Robin：

```python
hallrz.set_eb_robin(a0, b0, f0, lev=0)
hallrz.set_eb_robin(a1, b1, f1, lev=1)
hallrz.set_eb_robin(a2, b2, f2, lev=2)
```

要求：

- 每层 shape 为 `(Nr_lev, Nz_lev)`，对应该层 cell domain。
- 数组顺序为 WarpX RZ `(r,z)`，dtype 为 `np.float64`，C-contiguous。
- EB Neumann 存 `dphi/dn_EB`。
- EB Robin 存 `a*phi + b*dphi/dn_EB = f`。
- `n_EB` 是 computational/fluid region -> covered region。
- WarpX 不做 segment sign flip，也不把 coarse EB 数据插值到 fine EB 数据。

当前 AMR path 不支持 Python physical-domain boundary arrays。physical r-hi/z-hi/r-lo/z-lo 数据由
scalar/default 参数按每层网格生成。未来若增加 per-level physical arrays，也应由 Python 端按每层
真实 physical face node 直接给出，不应从 coarse physical BC 插值得到。

### 8.5 Benchmark 命令

用于复现本文 1/2/3 层速度与误差扫描的 driver：

```bash
source /home/shuliu/work/gpu_warpx2026/bin/activate
cd /home/shuliu/work/cuda_warpx_202604/WarpX
python3 Test/HallRZ/hallrz_stage4_8_amr_level_count_scan.py \
  --full-grids 448x512 672x768 896x1024 1120x1280 1344x1536 \
  --rmax 0.28 --zmax 0.32 \
  --steps 200 --chunk-steps 20 --max-grid-size 2048 \
  --three-target-ratio 6 --two-target-ratio 7 \
  --full-mcl 30 --two-mcls 30 30 --three-mcls 30 30 30 \
  --prefix hallrz_stage4_8_ratio6_200step
```

说明：

- 该 driver 使用 EB Neumann、physical scalar Robin/Dirichlet、static rho 和 fixed particles。
- 输出 CSV/MD 是最终汇总；JSON/NPZ/LOG 是 raw 复查材料，可在冻结后清理。
- 若使用自己的生产算例，应保留同样的误差统计口径：computational-region mask、AMR composite
  owned region、full-finest reference。

## 9. 调用路径与数据流

当前 HallRZ AMR Poisson 调用路径：

```text
WarpX::Evolve
  -> LabFrameExplicitES::ComputeSpaceChargeField
      -> DepositCharge / SyncRho
      -> ValidateHallRZAMRBoundaryData
      -> for lev = 0..max_level:
           HallRZPoissonSolver::ComputePhiAndE(...)
             -> ReadParameters()
             -> MLEBNodeFVLaplacian linop
             -> setDomainBC / setSigma / setRZ
             -> build rhs and physical level BC on this level
             -> fill EB Neumann or EB Robin data on this level
             -> setLevelBC(0, ...)
             -> setEBInhomogNeumann or setEBFVMRobin
             -> MakeNodalGeometry(...)
             -> setEBNodalGeometry(0, 0, finest geometry for this level solve)
             -> MLMG::solve(phi, rhs)
             -> ComputeStaggeredE(phi -> Er/Ez)
           if lev < max_level:
             InterpolateHallRZPhiBetweenLevels(coarse phi -> fine phi)
```

关键语义：

- 每个 AMR level 是一次独立 single-level EB-FVM solve；当前不是 composite AMR MLMG。
- 每层 physical domain BC 在该层 solve 内按该层 nodal grid 生成。
- fine patch 的 high-r/high-z artificial boundary 用当前 coarse phi Dirichlet 插值。
- `phi` 默认作为上一 PIC 步热启动保留；AMR 插值只刷新 artificial Dirichlet 边界和按权重混合
  fine patch 内部初始猜测。
- `ComputeStaggeredE` 写 Yee/staggered `Er/Ez`，并保护非计算区域/covered side，不得改成全 nodal
  诊断场。

## 10. 冻结保留文件

建议冻结时保留：

```text
Test/HallRZ/README_HallRZ_AMR_GRID_DESIGN_CN.md
Test/HallRZ/hallrz_stage4_8_amr_level_count_scan.py
Test/HallRZ/hallrz_stage4_8_small_ratio6_200step_amr_level_count_scan.{csv,md}
Test/HallRZ/hallrz_stage4_8_ratio6_200step_amr_level_count_scan.{csv,md}
Test/HallRZ/hallrz_stage4_8_ratio6_large_200step_amr_level_count_scan.{csv,md}
Test/HallRZ/hallrz_stage4_8_ratio5_representative_200step_amr_level_count_scan.{csv,md}
Test/HallRZ/hallrz_stage4_8_ratio7_representative_200step_amr_level_count_scan.{csv,md}
```

可以清理：

```text
Test/HallRZ/hallrz_stage4_8_*.log
Test/HallRZ/hallrz_stage4_8_*.json
Test/HallRZ/hallrz_stage4_8_*.npz
Test/HallRZ/hallrz_stage4_8_smoke_amr_level_count_scan.{csv,md}
```

保留 CSV/MD 的原因是它们包含速度、误差、最大误差位置和 pass/fail 状态，能复查本文结论。
raw JSON/NPZ/LOG 体积大、数量多，冻结后如无需逐点复算可以删除。
