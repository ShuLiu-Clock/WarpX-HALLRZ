# HallRZ EB Robin 重构与 AMR 分阶段计划

日期：2026-06-05

目标树：

```text
GPU WarpX: /home/shuliu/work/cuda_warpx_202604/WarpX
GPU AMReX: /home/shuliu/work/cuda_warpx_202604/AMReX
环境: /home/shuliu/work/gpu_warpx2026/bin/activate
```

本文件是 HallRZ EB Robin 重构的工作总计划。任何 compact context / 新 Codex 会话后，继续
本任务前必须先读本文件，再读 workspace/project `AGENTS.md`，然后才能继续设计、修改或测试。

## 0. 总目标与总门槛

本轮总目标分三条：

1. WarpX HallRZ Poisson solver 支持 EB Robin 边界，使用 AMReX 已重构的
   `MLEBNodeFVLaplacian` EB-FVM API。
2. HallRZ 几何允许 `lob == rmin`，此时通道从圆环退化为圆柱，低半径 EB wall 消失，低半径
   边界由物理 lo-r 边界或 RZ axis 处理。
3. 为未来最多三层 HallRZ AMR level-by-level Poisson solve 建立清晰代码框架，但第一批不直接
   实现三层 EB-FVM AMR 求解。

全局硬门槛：

- 每个 stage 内的小 phase 必须按顺序完成；每个 phase 的验收标准全部满足后，才能进入下一个
  phase。
- 每个 stage 完成后，必须通过约 1000 步 PIC 完整测试，并给出明确正确性与速度判断，才能进入
  下一个 stage。
- “能跑完不报错”不是验收标准。验收必须同时包含：
  - 无 NaN/Inf；
  - 物理/数值正确性指标达标；
  - 速度相对基准无不可解释退化；
  - HallRZ 特有几何、EB、Yee staggered E-field 保护检查通过。
- 不允许把未通过完整 PIC 验收的 stage 当作完成。
- 不允许在 Poisson solve 过程中创建或修改 AMR 网格。AMR 网格必须由 WarpX/AMReX 原生
  gridding 机制生成。
- HallRZ AMR 首版只支持静态三层 rectangular AMR：初始化后 `boxArray(lev)` 不能在运行中通过
  regrid 改变。运行中新增 level 或重划 box 的支持必须作为后续独立 stage 重新设计和验收。

方案修订规则：

- 本计划是当前证据下的工作设计，不是不可修改的真理。
- 后续施工中，如果代码审查、AMReX contract、数值测试、PIC 验收或性能数据提供了足够明确的
  证据，说明本计划某处设计不妥，必须停止继续按原方案推进。
- 不允许用局部补丁掩盖方案层面的错误；必须先报告证据，包括相关源码位置、测试命令、失败
  指标、日志摘要和可复现实验。
- 方案问题经讨论确认后，先更新本计划文档，再继续对应 phase。
- 若方案修订影响已经完成的 phase gate，受影响的 smoke、benchmark 或 1000 步 PIC gate 必须
  重新运行。

测试资源预授权：

- 本计划内的 HallRZ smoke、benchmark、stage gate PIC 测试允许直接运行，不需要每次再询问用户。
- CPU 测试允许直接运行，但总 CPU core 数不得超过 4。若使用 MPI 与 OpenMP，必须满足
  `np * OMP_NUM_THREADS <= 4`。
- CPU 多 MPI 并行测试必须覆盖，MPI rank 数不得超过 4。
- GPU 测试必须覆盖；默认使用 1 个 GPU。超过 1 个 GPU、生产规模算例或明显长作业仍需另行请求
  用户批准。
- 每个 stage gate 至少应包含 GPU PIC 完整测试和 CPU MPI PIC/或等价并行正确性测试。若某个测试
  因构建配置不可用，报告中必须说明原因，不能静默跳过。
- 2026-06-06 用户确认：当前 GPU tree HallRZ 阶段测试中，若 `np=4` 共享 1 张 GPU 的 MPI
  正确性测试结果正确，允许作为“多 MPI 下不出错”的并行 gate 通过。该测试目的只验证 MPI
  domain decomposition、field/particle ownership 和通信路径正确性，不作为速度基线。真实 CPU
  backend PIC gate 暂时延后，不阻塞当前 Stage 1。

## 1. 必须遵守的上下文恢复流程

每次 compact context 后，继续工作前按顺序执行：

1. 读本文件：

   ```bash
   bash -lc 'source /home/shuliu/work/gpu_warpx2026/bin/activate && sed -n "1,260p" /home/shuliu/work/cuda_warpx_202604/WarpX/HallRZ_EB_ROBIN_REFACTOR_PLAN_CN.md'
   ```

2. 读 workspace/project 指令：

   ```bash
   bash -lc 'source /home/shuliu/work/gpu_warpx2026/bin/activate && sed -n "1,220p" /home/shuliu/work/AGENTS.md && sed -n "1,220p" /home/shuliu/work/cuda_warpx_202604/WarpX/AGENTS.md'
   ```

3. 如涉及 AMReX solver contract，读：

   ```bash
   bash -lc 'source /home/shuliu/work/gpu_warpx2026/bin/activate && cd /home/shuliu/work/cuda_warpx_202604/AMReX && sed -n "1,360p" EB_FVM_REFACTOR_PLAN_CN.md && sed -n "1,180p" EB_FVM_EB_BC_UNIFIED_THEORY_CN.md'
   ```

4. 检查当前改动：

   ```bash
   bash -lc 'source /home/shuliu/work/gpu_warpx2026/bin/activate && cd /home/shuliu/work/cuda_warpx_202604/WarpX && git status --short'
   ```

5. 若发现已有用户或其他 agent 改动，不得回滚；先理解改动，再继续。

## 2. 禁止行为与关键注意事项

禁止行为：

- 不改 AMReX，除非 WarpX 接入暴露明确 AMReX bug，并先向用户报告。
- 不恢复旧实验开关：
  - `eb_fvm_rz_metric_mode`
  - `eb_fvm_rz_geom_weighted`
  - `robin_rhs_scale`
  - HallRZ 三整数 MG stop rule
- 不调用旧 `setEBDirichlet(Real)` 实现 EB-FVM Dirichlet/Robin。
- 不把 EB Dirichlet/Robin 实现成 covered-side neighbor flux。EB 永远是边界通量闭合。
- 不在 HallRZ Poisson solve 中创建 AMR `BoxArray` 作为正式网格。
- 不运行 MPI `np>4`，不运行生产 GPU 长作业，不 commit/push，除非用户明确批准。
- 不删除 build 目录，不执行 `git reset --hard`、`git clean`、`git restore`、`git checkout .` 等
  破坏性命令。
- 不把临时诊断留在最终代码中。临时诊断必须带
  `[SHULIU-CODEX-DIAG:<topic>:20260605]`，最终验收前必须搜索确认。

关键数值/实现注意事项：

- RZ 约定：WarpX 必须传 planar `(r,z)` nodal geometry；AMReX 内部负责乘 RZ metric。
- 数组顺序、参数命名、输入解析必须符合 WarpX 原生习惯和当前代码路径，不另立一套 HallRZ 私有
  规则。WarpX RZ 内部坐标/数组顺序按代码约定 `(r,z)`，不要按设计文档里的数学书写 `(z,r)`
  写接口数组。
- `r=0` 是 axis regularity boundary，不是 EB Robin。
- `lob == rmin` 时低半径 EB wall 不存在，不能在 EB geometry、EB BC、segment flux 统计中继续
  把 `r=lob` 当 EB wall。
- 物理边界和 EB 边界的方向约定必须严格区分。HallRZ 使用的是 AMReX nodal
  `MLEBNodeFVLaplacian`，其 raw `setLevelBC` 物理边界数据采用 nodal solver 的坐标正方向
  contract；EB-FVM 数据采用 EB normal contract。不得把二者混用。
- HallRZ 采用 Yee/staggered E-field。由 `phi` 计算 `Er/Ez` 时，非计算区域、covered 区域或跨
  EB covered side 的 staggered edge 必须保持保护：要么不计算，要么强制为 `0.0`。不得移除或
  弱化当前 “非计算区域不写有效电场” 的保护逻辑。
- `phi` solve 不得修改输入 `rho`。如临时缩放 RHS，必须用独立 RHS MultiFab。
- 每次 PIC 时间步的 HallRZ Poisson solve 必须使用上一 PIC 时间步已经收敛的 `phi` 作为初始猜测
  热启动。除非是初始化第一步、显式重置物理状态、或重建网格/场数据后没有合法旧解，不允许在
  每次 solve 前无条件 `phi.setVal(0.0)` 做冷启动。三层 AMR 路径中，coarse-to-fine 插值的
  `phi` 既是 fine level 初始猜测，也是人工边界 Dirichlet 数据来源；warm start 不得改变 RHS、
  EB BC 或 physical BC。
- HallRZ AMR 首版不得支持运行中 regrid 改变 `BoxArray`。若输入或运行路径会触发
  `boxArray(lev)` 变化，必须明确 abort 或禁用该路径，不能静默继续。
- AMReX 新 EB-FVM API 约定：
  - Neumann: `setEBInhomogNeumann(amrlev, eb_gn_cc)`
  - Dirichlet: `setEBFVMDirichlet(amrlev, eb_phi_cc)`
  - Robin: `setEBFVMRobin(amrlev, eb_a_cc, eb_b_cc, eb_f_cc)`
  - 数学形式：`a*phi + b*dphi/dn = f`

### 2.1 AMReX HallRZ BC 数据方向约定（已核对源码）

本节是后续 WarpX HallRZ 接入 AMReX EB-FVM 的固定 contract。compact context 后必须按本节
恢复边界方向，不得重新猜测。

已核对的 AMReX 位置：

- `AMReX_MLLinOp.H`: base interface 说明 data layout 和 derivative direction 由 derived operator
  定义；对 `MLNodeLinOp` / `MLEBNodeFVLaplacian` physical domain faces，`levelbcdata` 和
  `robinbc_[a|b|f]` 使用 face-resolved nodal components 和 positive coordinate derivative，不使用
  outward-normal derivative。
- `AMReX_MLNodeLinOp.H`: `setLevelBC` face component layout；
  `levelbcdata` 明确 stores `+x_i` direction, not outward-normal；`robinbc_a/b/f` 明确编码
  `a*phi + b*dphi/dx_i = f`，低/高 physical faces 都用同一个 positive coordinate derivative。
- `AMReX_MLNodeLinOp.cpp`: `setLevelBC` 只按 face component 复制 physical Robin `a,b,f`，不做
  低边界外法向翻转；ghost-fill/RHS 内部低/高边界符号与 positive-coordinate raw data 一致。
- `AMReX_MLEBNodeFVLaplacian.cpp`: physical Neumann/Robin RHS correction 使用
  positive-coordinate raw face data；EB-FVM flux records 使用 cell-centered `a,b,f`。
- `AMReX_MLEBNodeFVLaplacian.H`: EB Neumann normal is from computational region to covered region；
  EB-FVM Robin stores `a,b,f` in `a*phi + b*dphi/dn = f` on cut cells。
- `Tests/LinearSolvers/NodeEB_Neumann_Mix`: physical Robin test 构造为
  `a*phi + b*dphi/ds = f`，`ds` 是 coordinate-positive direction；EB Neumann 为
  fluid-to-covered normal。
- `Tests/LinearSolvers/NodeEB_HallRZ`: HallRZ EB segment normal 构造为 low-r `-e_r`、high-r
  `+e_r`、exit step `-e_z`。

物理 domain BC 的 raw `setLevelBC` 数据：

- RZ 中 AMReX coordinate 0 是 `r`，coordinate 1 是 `z`。不要按数学文档的 `(z,r)` 顺序组织
  MultiFab components。
- `setDomainBC(lobc, hibc)` 的数组顺序：
  - `lobc[0]`: r-lo，`hibc[0]`: r-hi；
  - `lobc[1]`: z-lo，`hibc[1]`: z-hi。
- `levelbcdata`、`robin_a`、`robin_b`、`robin_f` 的 face component 顺序在 2D RZ 为：

  ```text
  comp 0: r-lo
  comp 1: r-hi
  comp 2: z-lo
  comp 3: z-hi
  ```

- 物理 inhomogeneous Neumann `levelbcdata` 存的是坐标正方向导数，不是外法向导数：
  - comp 0 `r-lo`: `dphi/dr`
  - comp 1 `r-hi`: `dphi/dr`
  - comp 2 `z-lo`: `dphi/dz`
  - comp 3 `z-hi`: `dphi/dz`
- `r=0` axis 是 regularity boundary。AMReX RZ nodal path 对 axis inhomogeneous Neumann 无物理意义；
  HallRZ 不应把 axis 当成 EB Robin 或非零 inhomogeneous Neumann wall。
- 物理 Robin 的 `robin_a/b/f` 对 HallRZ nodal solver 同样按坐标正方向导数解释：

  ```text
  a*phi + b*dphi/ds = f
  ds = +r for r-lo and r-hi
  ds = +z for z-lo and z-hi
  ```

  因此若解析式给出 `dphi/dr`、`dphi/dz`，四个物理 face 都直接用
  `f = a*phi + b*dphi/dcoord`。不要在 r-lo 或 z-lo raw data 上额外乘负号。
- 如果上游物理模型或 WarpX 高层接口给出的是 physical domain outward-normal Robin：

  ```text
  a_out*phi + b_out*dphi/dn_out = f_out
  ```

  传给 AMReX nodal physical face raw `setLevelBC` 数据前必须转换为 positive-coordinate form：

  ```text
  lo face: a = a_out, b = -b_out, f = f_out
  hi face: a = a_out, b =  b_out, f = f_out
  ```

  在 2D RZ 中，lo face 包括 `r-lo` 和 `z-lo`，hi face 包括 `r-hi` 和 `z-hi`。只翻 Robin 的
  `b` 系数，不翻 `a`，不翻 `f`。同理，若上游给出 outward-normal Neumann
  `g_out = dphi/dn_out`，传入 `levelbcdata` 时 lo face 用 `-g_out`，hi face 用 `+g_out`。
- 上述 outward-normal 转换只适用于 physical domain face raw data。不要把这套 lo/hi 转换套到
  EB-FVM cut-cell BC 上。

EB-FVM BC 的 raw 数据：

- `setEBInhomogNeumann(0, eb_gn_cc)` 需要 cell-centered cut-cell data，数学含义是
  `dphi/dn = eb_gn_cc`。
- EB normal `n` 的方向是 AMReX 明确约定的 computational/fluid region -> covered region。
- `setEBFVMRobin(0, eb_a_cc, eb_b_cc, eb_f_cc)` 的数学形式是
  `a*phi + b*dphi/dn = f`，这里的 `n` 仍然是 fluid-to-covered EB normal。
- `setEBFVMDirichlet(0, eb_phi_cc)` 等价于 `a=1,b=0,f=phi_eb`，但必须调用新 EB-FVM API，不走旧
  covered-node Dirichlet。
- EB `a,b,f`、`gn` 不带额外 `1/r`，也不要传 RZ-preweighted geometry。WarpX 传 planar `(r,z)`
  几何，AMReX 内部负责 RZ metric。
- HallRZ EB segment normal 约定：

  ```text
  segment 0: r = lob, z in [zlo,out], n = -e_r, only exists when lob > rmin
  segment 1: r = hib, z in [zlo,out], n = +e_r
  segment 2: z = out, r in [rmin,lob], n = -e_z, only exists when lob > rmin
  segment 3: z = out, r in [hib,rmax], n = -e_z
  ```

  `lob == rmin` 时 segment 0 和 segment 2 长度为零，不能继续生成 EB BC 或 flux 统计。

### 2.2 WarpX/Python 边界数据接口设计准则

本项目在 WarpX 端硬性规定：Python 端传入的 HallRZ 边界数据必须已经满足 AMReX raw
`MLEBNodeFVLaplacian` 设计要求。WarpX 不在默认接口里猜测、识别或自动转换另一套方向口径。

设计准则：

- Python API 文档、参数名、错误信息都必须写明数据方向采用 AMReX raw contract。
- 对 physical domain face 数据，如果未来允许 Python 传入 `levelbcdata` 或 physical Robin
  `a,b,f`，Python 端传入值必须已经是 positive-coordinate derivative 口径：
  - r-lo/r-hi 都是 `dphi/dr`；
  - z-lo/z-hi 都是 `dphi/dz`；
  - physical Robin raw form 是 `a*phi + b*dphi/dx_i = f`。
- 若用户物理模型使用 outward-normal Robin/Neumann，必须在调用 WarpX Python API 之前按
  Section 2.1 的 lo/hi 规则转换成 AMReX raw face data。当前设计不增加
  `normal_convention=outward`、`robin_direction=...` 这类可选开关。
- 对 EB-FVM cut-cell 数据，Python 端传入的 `gn` 或 Robin `a,b,f` 必须已经使用
  `n_EB = fluid/computational region -> covered region`：
  - `set_eb_neumann` 等价数据是 `dphi/dn_EB`；
  - `set_eb_robin(a,b,f)` 的 `f` 必须已经满足 `a*phi + b*dphi/dn_EB = f`；
  - WarpX 不根据 segment 编号自动翻转用户传入的 EB `gn` 或 Robin `b/f`。
- WarpX 端只负责 shape、dtype、finite、cut-cell coverage、`abs(a)+abs(b)>0` 等结构性校验；
  方向正确性属于 Python 调用方与测试算例必须满足的物理 contract。
- 如果后续确实需要 outward-normal 或 segment-local convenience API，必须另起明确命名的高层
  wrapper，并在 wrapper 内显式转换到本节的 AMReX raw contract；不能改变现有 raw API 的语义。

### 2.3 WarpX/AMReX AMR 与 MPI ownership 设计准则

HallRZ AMR 必须延续 WarpX/AMReX 原生多层网格和 MPI 管理模型，不另立一套 rank/block 规则。

已核对的 WarpX/AMReX 设计要点：

- 每个 AMR level 由一组三元数据定义：
  - `Geom(lev)`：该 level 的完整物理/index domain；
  - `boxArray(lev)`：该 level 当前有效网格 boxes；
  - `DistributionMap(lev)`：`boxArray(lev)` 中每个 box 的 MPI owner rank。
- `DistributionMapping::ProcessorMap()[i]` 表示第 `i` 个 box 所属的 MPI rank。一个 MPI rank 可以
  同时拥有多个 level 上的多个 boxes，但一个 box/FAB/MultiFab 不跨越多个 AMR level。
- 不能假设同一物理区域的 coarse box 和 fine box 在同一 MPI rank 上。跨 level 数据传递必须使用
  MultiFab/AMReX 通信接口，例如 `ParallelCopy`、coarse/fine interpolation、`FillBoundary` 或
  `SumBoundary`。
- 一个逻辑 rectangular AMR patch 可能被 `max_grid_size`、blocking factor、`ChopGrids` 或 load
  balance 切成多个 boxes。由这些机制产生的 box 内部分割面只是 FAB/MPI 分块边界，不是物理
  boundary，也不是 AMR artificial boundary。
- HallRZ solver 临时 MultiFab、RHS、BC mask、artificial boundary value、EB BC data 必须使用目标
  level 的场数据 `boxArray()/DistributionMap()` 或 `warpx.boxArray(lev)/warpx.DistributionMap(lev)`。
  不得临时构造一个新的 `DistributionMapping(level_ba)` 来参与已有场数据求解，除非该数据本身是
  完全独立的 temporary，并且不会与现有 level MultiFab 直接混用。
- 粒子容器默认通过 AMReX `AmrParGDB` 使用 `AmrCore::boxArray/DistributionMap`。若未来某处显式
  设置了 particle 专用 `ParticleBoxArray/ParticleDistributionMap`，HallRZ field solver 仍应以场
  MultiFab 的 `boxArray/DistributionMap` 为准，不使用 particle dmap 作为 field solve dmap。
- WarpX load balance 可在 `boxArray(lev)` 不变时更换 `DistributionMap(lev)`，并通过
  `RemakeLevel`、field `Redistribute` 和 particle `Redistribute` 迁移数据。HallRZ solver 不应缓存
  过期 dmap；每次 solve 从当前 WarpX field MultiFab/AmrCore 读取 ba/dm。
- 现有普通 Poisson 路径的 `interpolatePhiBetweenLevels` 是 coarse-to-fine 数据流参考：先在 fine
  patch coarsened `BoxArray` 和 fine level `DistributionMap` 上创建临时 coarse MultiFab，再从
  coarse level `phi` 做 `ParallelCopy`，最后本地插值到 fine `phi`。HallRZ AMR 不得按 rank 本地
  指针或裸数组假设跨层数据同址。
- 判断 physical boundary 与 patch boundary 时，参考 WarpX PML 初始化逻辑：
  `boxArray(lev).minimalBox()` 与 `Geom(lev).Domain()` 重合的 face 才可能继承 physical domain BC；
  位于物理 domain 内部的 patch face 才是 AMR artificial boundary。

HallRZ AMR 首版固定限制：

- 只支持初始化时生成的静态三层 lower-left rectangular AMR layout，`max_level == 2`，相邻
  refinement ratio 为 2。
- 不支持运行中 regrid 改变 `boxArray(lev)`，不支持运行中新增 level，不支持动态图形 tag 改变
  Ω1/Ω2。
- 运行中 load balance 只要不改变 `boxArray(lev)`、只改变 `DistributionMap(lev)`，原则上应兼容；
  但 HallRZ solver 必须每次从当前 MultiFab/AmrCore 读取 ba/dm，不能缓存初始化时的 dmap。
- AMR artificial boundary 是 level valid region 的边界概念，不是 MPI rank 边界、不是 box 内部分割
  边界。

## 3. Stage 0：基准测试与测试框架先行

Stage 0 是所有代码重构前的最早工作。目的不是改 solver，而是先固定对照组、完整 PIC 算例、
正确性指标和速度基准。后续每个 stage 都必须复用这个基准体系。

### Phase 0.1：记录当前对照组状态

目标：

- 在任何 HallRZ solver 源码重构前，记录当前 GPU WarpX tree 状态。
- 明确“baseline”是当前重构开始前的 WarpX 程序，而不是重构后的程序。

任务：

- 记录 `git rev-parse HEAD`。
- 记录 `git status --short`。
- 记录已有 HallRZ 相关 dirty files，区分用户已有改动和后续 Codex 改动。
- 不回滚任何已有改动。

验收标准：

- 有一段 baseline 记录，包含 commit hash、dirty status、测试环境、构建目录。
- 用户能从记录判断后续速度对比的对照程序是哪一个。

### Phase 0.2：编写 PIC 速度/正确性对照算例

目标：

- 在 solver 重构前建立一个约 1000 步 HallRZ PIC benchmark。
- 该 benchmark 后续作为所有 stage 的速度和正确性对照。

建议文件：

```text
Test/HallRZ/picmi_hall_rz_pic_benchmark.py
Test/HallRZ/README_HallRZ_EB_Robin_Refactor_Benchmark.md
```

算例要求：

- RZ HallRZ electrostatic PIC。
- 使用当前已工作的 EB Neumann path，作为 baseline。
- `max_step` 约 1000，除非用户批准不同步数。
- 必须包含 GPU baseline，默认 `np=1`、1 GPU。
- 必须包含 CPU 多 MPI baseline，`np<=4` 且总 CPU core 数不超过 4；本计划内这些测试不需要
  再询问用户。
- 固定随机种子、粒子数、网格、时间步长、诊断间隔。
- 输出足够少，不能让 I/O 主导速度。
- 记录 solver wall time、step wall time、MLMG iter、残差、NaN/Inf、粒子数、电荷统计。

正确性指标建议：

- `phi`、`Er`、`Ez` 全程无 NaN/Inf。
- `rho` 不被 Poisson solve 修改：`rho_mutation_inf == 0` 或低于 roundoff 级别。
- EB flux diagnostic 中 `q_EB_added_RZ` 与 `q_EB_total_RZ` 相对误差在既有 Neumann path 的
  baseline 容差内。
- 粒子数、总电荷、域内/壁面损失统计随时间无非物理跳变。
- HallRZ staggered E-field covered 区域保护抽查通过：非计算区域 `Er/Ez` 不产生有效非零场。

速度指标建议：

- 记录总 1000 step wall time。
- 记录平均每 step wall time。
- 记录 Poisson solve 平均 wall time 与 p50/p95。
- 记录平均 MLMG iter。
- 后续 stage 速度不得比 baseline 慢超过 10%，除非有明确物理功能增加且用户接受。

验收标准：

- benchmark 文件能运行。
- baseline 1000 步完成。
- correctness metrics 全部输出并达标。
- 速度基准形成固定表格，后续 stage 必须对照。

### Phase 0.3：建立快速 smoke 测试

目标：

- 提供每次小改后的快速反馈，不替代 1000 步完整 PIC gate。

任务：

- 基于 benchmark 派生 1-20 步 smoke。
- smoke 检查编译、Python API、无 NaN/Inf、基本 solve 调用路径。

验收标准：

- smoke 可在短时间运行。
- smoke 输出与 benchmark 使用同一套关键字段。
- 文档明确：smoke 只能作为 phase 内快速检查，不能作为 stage 完成 gate。

### Stage 0 完成门槛

必须满足：

- benchmark 已经在当前未重构 solver 上完成约 1000 步。
- correctness metrics 达标。
- baseline 速度表已记录。
- 用户审查 benchmark 与 baseline 结果并批准进入 Stage 1。

## 4. Stage 1：单层 HallRZ 重构、EB Robin、`lob == rmin`

Stage 1 只处理单 AMR level。`LabFrameExplicitES` 中 HallRZ path 仍要求 `max_level == 0`。

### Phase 1.1：拆分 HallRZ 代码结构

目标：

- 从“单大文件 + 混杂职责”重构为小模块，便于后续维护。

建议模块：

```text
Source/FieldSolver/ElectrostaticSolvers/HallRZPoissonSolver.H
Source/FieldSolver/ElectrostaticSolvers/HallRZPoissonSolver.cpp
Source/FieldSolver/ElectrostaticSolvers/HallRZGeometry.H
Source/FieldSolver/ElectrostaticSolvers/HallRZGeometry.cpp
Source/FieldSolver/ElectrostaticSolvers/HallRZBoundaryData.H
Source/FieldSolver/ElectrostaticSolvers/HallRZBoundaryData.cpp
Source/FieldSolver/ElectrostaticSolvers/HallRZSingleLevelSolve.H
Source/FieldSolver/ElectrostaticSolvers/HallRZSingleLevelSolve.cpp
```

职责边界：

- `HallRZGeometry`：参数、几何验证、segment 存在性、planar nodal geometry。
- `HallRZBoundaryData`：Python/C++ EB Neumann/Robin/Dirichlet 数据存储、shape/finite 校验。
- `HallRZSingleLevelSolve`：构造 linop、RHS、level BC、EB BC、调用 MLMG、写回 E-field。
- `HallRZPoissonSolver`：保留外部 API facade，减少调用方改动。

验收标准：

- 编译通过。
- 原有单层 EB Neumann smoke 通过。
- `git diff` 显示职责拆分清楚，无 unrelated refactor。
- `ComputeStaggeredE` 的 covered/非计算区域保护逻辑仍存在并有清晰单元边界。

### Phase 1.2：接入 AMReX 新 EB-FVM API

目标：

- WarpX 只提供 finest-level HallRZ geometry 和 finest-level EB BC。
- AMReX 负责 MG-level geometry、BC coarsening、EB flux records。

任务：

- 删除或停用 HallRZ 手动循环上传所有 MG geometry 的路径。
- 调用 `linop.setEBNodalGeometry(0, 0, eb_node_geom_finest)`。
- Neumann mode 调用 `linop.setEBInhomogNeumann(0, eb_gn_cc)`。
- 不再使用旧 RZ metric/weighted experimental switches。
- 不使用 HallRZ 三整数 MG stop rule。

验收标准：

- `rg -n "setEBNodalGeometry|setEBFVMRobin|setEBInhomogNeumann|setEBDirichlet|RZMetric|rz_weighted|AlignedMaxCoarsening" Source/FieldSolver/ElectrostaticSolvers`
  的结果符合新 contract。
- AMReX handoff 中禁止恢复的旧开关未出现。
- EB Neumann smoke 与 baseline 对比，结果在容差内。

### Phase 1.3：实现单层 EB Robin

目标：

- HallRZ EB-FVM 支持 Python 提供 cell-centered `a,b,f`。

接口：

```python
hallrz.set_eb_robin(a, b, f, lev=0)
hallrz.clear_eb_robin(lev=0)
hallrz.has_eb_robin(lev=0)
```

Stage 1 限制：

- 只允许 `lev=0`。
- shape 必须按 WarpX RZ 原生数据习惯设计并在接口文档中明确；当前 cell-centered 全局 EB 数组
  采用 `(Nr,Nz)`，索引顺序为 `(r,z)`。
- dtype 必须 `np.float64`，C-contiguous。

任务：

- 新增 `warpx.hall_rz_eb_bc_mode = neumann|robin`，默认 `neumann`。
- Robin mode 下调用 `linop.setEBFVMRobin(0, eb_a_cc, eb_b_cc, eb_f_cc)`。
- Robin mode 下若 Python EB Robin 缺失，abort。
- Robin mode 下若旧 scalar wall flux 或 Python EB Neumann active，abort。
- cut-cell 上检查 finite、`abs(a)+abs(b)>0`。
- Python 端 `set_eb_robin(a,b,f)` 必须作为 AMReX raw EB-FVM API 暴露，硬性要求用户传入
  `n_EB = fluid/computational region -> covered region` 口径的数据；WarpX 不按 segment 自动翻转。
- 在代码注释或测试文档中明确 EB Robin 的 `dphi/dn` 符号采用 AMReX EB-FVM
  fluid-to-covered normal；物理 domain Robin 的 raw `setLevelBC` 数据采用 AMReX nodal solver
  的 coordinate-positive derivative 约定。两者不得混用。

验收标准：

- EB Robin smoke 通过。
- 对 Neumann 等价 Robin，即 `a=0,b=1,f=gn`，结果与 Neumann baseline 在预设容差内一致。
- 对 Dirichlet 等价 Robin，即 `a=1,b=0,f=phi_eb`，能走 `setEBFVMRobin` 而不是旧 covered-node
  Dirichlet path。
- 至少一个带符号敏感的 manufactured/解析 Robin 检查通过，用于确认物理 domain raw
  coordinate-positive derivative 和 EB fluid-to-covered normal 没有被混淆或重复翻转。
- 无 NaN/Inf，MLMG 收敛记录正常。

### Phase 1.4：支持 `lob == rmin`

目标：

- 允许圆柱退化几何，低半径 EB wall 消失。

任务：

- 几何验证从 `rmin < lob < hib < rmax` 改为 `rmin <= lob < hib < rmax`。
- 当 `lob == rmin`：
  - EB2 不构造 low covered block；
  - segment 0 和 segment 2 标记为不存在；
  - `hallSegment`、EB Neumann/Robin 填充、EB flux stats 不消费消失 segment；
  - `eb_wall_flux_segment=0` 或 `2` 时 abort；
  - 低半径边界由 physical lo-r BC 或 axis 处理。
- 当 `lob > rmin`：保持原圆环几何行为。

验收标准：

- `rmin=0,lob=0` 圆柱 axis smoke 通过。
- `rmin>0,lob=rmin` 圆柱 physical lo-r BC smoke 通过。
- 原圆环 `lob>rmin` benchmark/smoke 不回退。
- EB flux segment 输出只包含实际存在的 segments。

### Phase 1.5：Stage 1 MG/MCL 性能恢复补充 gate

目标：

- 解除 Stage 1 当前速度 blocker，并把 HallRZ MG 最大层数策略写成可维护接口，而不是临时补丁。
- WarpX C++ 不做自动性能启发式，不恢复 HallRZ 三整数 MG stop，也不把 `lob/hib/out` alignment
  规则塞回 AMReX。
- WarpX 只从 Python/ParmParse 接收 `warpx.hall_rz_max_coarsening_level`，将其作为 AMReX
  `LPInfo::setMaxCoarseningLevel()` 的 upper bound。AMReX 继续负责实际 MG-level 生成、cut-EB
  admissibility stop 和 `NMGLevels(0)`。
- 通过矩阵 benchmark 让 Python/user 侧选择目标算例的最优 MCL。C++ 不 hardcode `MCL=0` 或其它
  经验值。
- 修正单层 solve 初始猜测语义：每个 PIC 时间步默认使用上一 PIC 时间步的 `phi` 热启动，不允许
  每步冷启动；第一步或显式 reset 例外。

任务：

- benchmark driver 增加显式参数：

  ```text
  --hall-rz-max-coarsening-level <int>
  ```

  并在 JSON summary 中记录：

  ```text
  requested_hall_rz_max_coarsening_level
  ```

- C++ HallRZ 诊断在 `params.verbose >= 1` 或 Poisson summary 行中记录：

  ```text
  requested_mcl=<value>
  actual_nmg_levels=<linop.NMGLevels(0)>
  ```

  若 AMReX 因 EB-FVM RZ cut stop 或 semicoarsening 保留/裁剪 MG levels，日志必须能从
  `requested_mcl` 和 `actual_nmg_levels` 看出来。
- `ComputePhiAndE()` 不得在每步 solve 前无条件清零 `phi`。若需要初始化第一步的 `phi`，必须只在
  没有合法旧解时做一次性初始化，或依赖 WarpX field 初始化后的已有值。后续 PIC steps 直接把
  当前 `phi` 传给 `MLMG::solve()` 作为 initial guess。
- 保留现有 `rho` mutation 检查和 staggered `Er/Ez` covered-side 保护；本 phase 不改变
  `ComputeStaggeredE()` 的保护逻辑。
- 跑受控 MCL 矩阵：

  ```text
  MCL = 0
  MCL = 1
  MCL = 2
  MCL = 当前默认/用户请求值
  ```

  Neumann 必须全跑。Neumann-equivalent Robin 至少跑 `MCL=0` 和当前默认/用户请求值；若二者速度
  或迭代行为不一致，再补齐 `MCL=1,2`。
- 每个矩阵点记录：
  - `avg_step_wall_time_s`、`total_step_wall_time_s`、p50/p95 chunk time；
  - HallRZ Poisson `Final Iter` 平均值/最大值；
  - final residual 范围；
  - `Fapply/Fsmooth/MLMG::solve/prepareForSolve` profiler 摘要；
  - `requested_mcl`、`actual_nmg_levels`；
  - `phi/Er/Ez/rho` finite、min/max/max_abs/l2；
  - `N_NaN/N_Inf`、粒子 NaN/Inf、粒子数。

验收标准：

- `--hall-rz-max-coarsening-level` 能从 Python benchmark 传到 WarpX C++，再传到 AMReX
  `LPInfo::setMaxCoarseningLevel()`。
- 日志能明确显示 requested MCL 与 AMReX actual NMG levels；不能只能靠手工推断。
- `MCL=0`、`MCL=1`、`MCL=2`、默认/用户请求值的 Neumann 矩阵全部完成，且 JSON/log 可复查。
- Neumann-equivalent Robin 与 Neumann 在相同 MCL 下的速度和场量保持等价；若不等价，Stage 1
  不能通过。
- 若某个显式 MCL 恢复到 Stage 0 baseline 的 10% 容差内，并且正确性指标全部达标，则该 MCL 可
  作为当前单层 Stage 1 gate 的推荐运行设置。推荐设置必须写入测试报告，不写入 C++ hardcode。
- 热启动实现后，MLMG 迭代数、残差和场量必须正常；不能因为复用旧 `phi` 导致 stale solution、
  0-iter 假快或 BC/RHS 未更新。
- 若所有 MCL 都慢于 Stage 0 超过 10%，必须先给出 profiler 归因和正确性/速度 tradeoff 报告，
  由用户决定是否放宽 Stage 1 速度门槛；不得直接进入 Stage 2。

禁止：

- 不恢复旧 HallRZ 三整数 MG stop rule。
- 不恢复 `eb_fvm_rz_metric_mode`、`eb_fvm_rz_geom_weighted`、`robin_rhs_scale` 等旧实验开关。
- 不在 AMReX 中 hardcode HallRZ channel 几何或 MCL policy。
- 不把 `MCL=0` 写成 WarpX C++ 默认性能策略；只能由 Python/ParmParse 用户显式设置。

### Phase 1.6：Stage 1 完整 PIC 验收

目标：

- 证明重构后的单层 HallRZ 在真实 PIC 运行中正确且速度可接受。

必须运行：

- 使用 Phase 1.5 推荐显式 MCL 的 Stage 0 benchmark baseline-equivalent 1000 步 run。
- 使用同一显式 MCL 的 EB Robin 1000 步 run。
- `lob == rmin` 1000 步 run。若 GPU 时间紧张，可先选 `rmin=0,lob=0`，但进入 Stage 2 前必须
  补齐 `rmin>0,lob=rmin`。
- GPU 与 CPU 多 MPI 都必须覆盖。CPU 总 core 数不超过 4，MPI rank 数不超过 4。

正确性验收：

- 无 NaN/Inf。
- `rho` 不被 Poisson solve 修改。
- solve 使用上一 PIC 时间步 `phi` 热启动；诊断不能显示 stale RHS/BC 或 0-iter 假快。
- Neumann 等价 Robin 与 Neumann baseline 差异在容差内。
- EB flux/RHS diagnostic 一致性达标。
- 粒子数、电荷、壁面损失统计无非物理跳变。
- covered/非计算区域 staggered `Er/Ez` 保护通过。

速度验收：

- 1000 步总 wall time 不慢于 Stage 0 baseline 的 10% 以上，除非用户批准。
- 平均 Poisson solve time、MLMG iter 无异常增长。
- 若速度退化，必须定位到具体 phase 或功能开销，不能直接进入 Stage 2。

Stage 1 完成门槛：

- 所有 Phase 1.1-1.6 验收通过。
- 用户审查 Stage 1 完整 PIC 报告并批准进入 Stage 2。

## 5. Stage 2：HallRZ 专用 AMR 初始化网格框架

Stage 2 只建立 HallRZ 三层 AMR 网格生成和验证框架，不实现 EB-FVM AMR Poisson solve。
首版采用静态 rectangular AMR：Ω1/Ω2 在初始化时确定，运行中不允许 regrid 改变 boxes。

本阶段和 Stage 3 采用本计划定义的三层 lower-left nested AMR patch 准则。此前 HallRZ
MG-level face-alignment、三整数 MG stop rule、手动上传所有 MG 几何、旧 nonaligned coarse
geometry 开关等规则不能作为 AMR level 边界设计依据。
同时，本阶段和 Stage 3 必须遵守 Section 2.3 的 WarpX/AMReX AMR/MPI ownership 设计准则。

### Phase 2.1：HallRZ AMR 参数接口

目标：

- 用 HallRZ 专用参数表达用户设计的 Ω1/Ω2。

接口原则：

- 参数使用具名 `r1,z1,r2,z2`，避免裸数组顺序歧义。
- 代码内部 RZ 顺序仍是 `(r,z)`。
- AMR ratio 使用 WarpX/AMReX 原生 `amr.ref_ratio = 2`。

建议参数：

```text
warpx.hall_rz_amr_enable = 1
warpx.hall_rz_amr_r1 = ...
warpx.hall_rz_amr_z1 = ...
warpx.hall_rz_amr_r2 = ...
warpx.hall_rz_amr_z2 = ...
```

MG upper-bound 参数设计：

- 单层 HallRZ 继续使用既有 scalar 参数：

  ```text
  warpx.hall_rz_max_coarsening_level = M
  ```

- 三层 HallRZ AMR 后续增加可选 per-AMR-level 参数：

  ```text
  warpx.hall_rz_max_coarsening_levels = M0 M1 M2
  ```

  顺序固定为 AMR level 0/1/2，不是 `(r,z)`，也不是 AMReX MG level。若该向量缺失，则把 scalar
  `warpx.hall_rz_max_coarsening_level` 广播到所有 AMR levels。
- WarpX C++ 只把每个 AMR level 对应的 `Mlev` 作为 `LPInfo::setMaxCoarseningLevel(Mlev)` 传给
  AMReX。AMReX 继续决定该 level 实际保留多少 MG levels。
- 不允许 WarpX C++ 在三层 AMR 中自动试探或 hardcode 每层最优 MCL。每层最优值由 Python/user
  通过 benchmark 决定。

验收标准：

- 参数解析与错误信息清楚。
- `max_level` 必须为 `0` 或 `2`；Stage 2 若启用 AMR 框架但 solver 仍单层，必须明确 abort 或
  只做 grid validation 模式。
- `RefRatio(0)==RefRatio(1)==2`。
- scalar MCL 与 per-level MCL 向量的优先级、长度和错误信息明确；向量长度不是 3 时必须 abort。
- 运行中 regrid 改变 `boxArray` 的路径被禁止或明确 abort。

### Phase 2.2：level-aware tagging

目标：

- 复用 WarpX/AMReX 原生初始化 gridding，在 `WarpX::ErrorEst(lev,...)` 里加入 HallRZ 专用
  level-aware tag rule。

规则：

- `lev==0` tag Ω1：`r in [rmin,r1]` 且 `z in [zmin,z1]`。
- `lev==1` tag Ω2：`r in [rmin,r2]` 且 `z in [zmin,z2]`。
- 不支持 `lev>=2` tagging。
- 禁止同时使用 `warpx.fine_tag_lo/hi` 或 `warpx.ref_patch_function` 作为 HallRZ AMR source。

验收标准：

- 初始化后 `finestLevel()==2`。
- `boxArray(1)` 覆盖 Ω1，`boxArray(2)` 覆盖 Ω2；允许 AMReX 因 `max_grid_size` 切成多个 boxes。
- 多个 boxes 只表示 AMReX/WarpX 原生 gridding/MPI decomposition，不改变 Ω1/Ω2 的物理含义。
- boxes 满足 proper nesting、blocking factor、max_grid_size。
- 该阶段仍不进入 HallRZ 多层 Poisson solve。

### Phase 2.3：AMR geometry validation

目标：

- 验证用户嵌套网格设计符合 HallRZ 几何约束。

验证：

- `zmin < out < z2 < z1 < zmax`。
- `rmin <= lob < hib < r2 < r1 < rmax`。
- `lob/hib/out/r1/r2/z1/z2` 全部落在对应 level cell face 上。
- `lob == rmin` 时 low EB wall 消失规则仍成立。
- channel 完整包含在 Ω2 内。

验收标准：

- 不合法输入给出明确 abort message。
- 合法输入生成预期 AMR levels。
- 与单层 HallRZ benchmark 的默认配置互不影响。

### Phase 2.4：Stage 2 完整 PIC 验收

目标：

- 验证新增 AMR 初始化框架没有破坏单层 HallRZ 生产路径。
- 如 Stage 2 暂不允许 AMR solve，则 PIC 完整测试仍用 `max_level=0` 跑 Stage 1 benchmark，同时
  用短初始化测试验证 AMR grids。

必须运行：

- Stage 1 benchmark 1000 步，确认单层性能和正确性不回退。
- HallRZ AMR grid initialization smoke，确认 `finestLevel()==2` 和 boxes 正确。
- 静态 AMR 限制 smoke：确认 HallRZ AMR 模式下不会触发运行中 `boxArray` 改变；若触发则明确
  abort。
- GPU 与 CPU 多 MPI 都必须覆盖。CPU 总 core 数不超过 4，MPI rank 数不超过 4。

进入 Stage 3 门槛：

- 单层 1000 步 PIC 完整测试仍达标。
- AMR grid initialization 验证通过。
- 用户审查并批准进入 Stage 3。

## 6. Stage 3：HallRZ level-by-level AMR Poisson solve

Stage 3 才实现三层 HallRZ AMR Poisson solve。该阶段风险最高，必须小步推进。首版仍只支持
静态三层 rectangular AMR，不支持运行中 regrid 改变 `boxArray`。

### Phase 3.1：数据流与 driver skeleton

目标：

- 在 `LabFrameExplicitES` HallRZ path 中支持 `max_level<=2` 的 driver skeleton。

数据流：

- `rho_fp[lev]` 来自 WarpX/AMReX 原生 deposition 和 `SyncRho`。
- 按 `lev=0 -> 1 -> 2` 顺序 solve。
- 每层使用 WarpX/AMReX 原生生成的该 level `Geom/BoxArray/DistributionMapping/EBFactory`，
  不在 solve 阶段创建新的 AMR grid 或把 MG level 当 AMR level。
- 每次 solve 从当前 `rho_fp[lev]`、`phi_fp[lev]` 或 `warpx.boxArray(lev)/DistributionMap(lev)`
  读取 ba/dm，不缓存初始化时的 `DistributionMapping`。
- 每个 AMR level 独立构造一个 AMReX linop/MLMG solve，并使用该 AMR level 对应的
  `hall_rz_max_coarsening_levels[lev]` 作为 AMReX MG upper bound。不得把 AMReX MG level 当成
  WarpX AMR level，也不得让三层 AMR 共用一个无法覆盖各层差异的隐式最优 MCL。
- Level 0 在完整物理 domain 上求解。Level 1/2 在当前 AMR patch 有效区域上求解；若一个 level
  被 `max_grid_size` 切成多个 boxes，这些 boxes 的内部切分边界只是 FAB/MPI 分块边界，不是
  物理边界，也不是 coarse Dirichlet 边界。
- 解完 coarse level 后，将 `phi[lev]` 插值到 `phi[lev+1]`：
  - 作为下一层 `phi` 的初始猜测解，用于 MLMG warm start；
  - 作为下一层 AMR 人工边界的 Dirichlet 边界值。
- coarse-to-fine 插值必须走 MultiFab/AMReX 并行通信路径，参考普通 Poisson
  `interpolatePhiBetweenLevels`。不得按 MPI rank 本地裸数组假设 coarse/fine 同 rank 或同 box。
- warm start 不能改变 RHS、EB BC、physical domain BC 或人工边界分类；它只影响初值。

验收标准：

- `max_level==0` 路径不回退。
- `max_level==2` 可进入 driver skeleton 并在未实现 solve 的阶段明确 abort，不静默给错结果。
- driver 日志能区分每层 physical boundary、AMR artificial boundary、EB boundary 和内部 box
  分块边界。
- driver 日志能逐 AMR level 打印 requested MCL 和该 level 的 actual NMG levels。
- 若运行路径试图 regrid 改变 `boxArray` 或新增 level，HallRZ AMR path 明确 abort，不继续求解。

### Phase 3.2：fine patch artificial boundary

目标：

- 实现 Ω1/Ω2 因 AMR 分层产生的人工边界 coarse Dirichlet 处理。

规则：

- 只有 AMR 分层产生的人工边界使用 coarse Dirichlet。
- 与物理 domain 边界重合的 patch face 必须继续使用物理边界条件，不允许改用 coarse
  Dirichlet。对当前 lower-left nested layout，`lo-r`、`lo-z` 与物理 domain 低边界重合时使用
  physical lo-r/lo-z BC。
- patch 的 `hi-r`/`hi-z` 若位于物理 domain 内部，是 AMR artificial boundary，使用上一层
  coarse `phi` 插值后的 Dirichlet。
- 如果未来某个 patch face 与物理 `r-hi` 或 `z-hi` domain 边界重合，该 face 也必须使用对应
  physical hi-r/hi-z BC，而不是 coarse Dirichlet。
- EB 边界永远使用 EB-FVM BC，不允许把 EB boundary 当成 coarse Dirichlet、open boundary 或
  wall-neighbor flux。
- 同一 AMR level 内由 `max_grid_size`、blocking factor 或 MPI decomposition 造成的 box 内部分割
  不是边界；不得在这些内部 box face 上施加 coarse Dirichlet 或 physical BC。
- 一个 MPI rank 可以同时拥有多个 level 的 boxes；这个 rank ownership 事实不能用于定义人工
  边界，也不能用于跳过跨 level `ParallelCopy`/插值通信。
- 若当前 AMReX nodal operator API 不能直接表达“patch 内部人工边界 Dirichlet”，必须在
  Phase 3.1/3.2 明确提出实现机制和证据，不能把人工边界伪装成 physical domain boundary 后
  静默推进。

验收标准：

- 人工边界节点值来自上一层 coarse interpolation。
- 细层 `phi` 初始猜测来自上一层 coarse interpolation，MLMG solve 以该初值 warm start。
- physical boundary nodes 不使用 coarse Dirichlet；人工边界 nodes 不使用 physical BC。
- 内部 box 分块边界没有被计入 artificial boundary masks。
- 不与 EB boundary flux 重复计入。
- 对无 EB 或简单解析源 case，level-by-level 结果与单层/解析解趋势一致。

### Phase 3.3：per-level EB geometry 与 EB Robin data

目标：

- 每个 AMR level 独立构造该 level 的 HallRZ planar nodal EB geometry 和 EB BC。

接口：

```python
hallrz.set_eb_robin(a, b, f, lev=0)
hallrz.set_eb_robin(a, b, f, lev=1)
hallrz.set_eb_robin(a, b, f, lev=2)
```

规则：

- Python 逐 level 提供 cell-centered `a,b,f`。
- WarpX 只做 shape、finite、cut-cell 校验和上传。
- 不从 finest 自动 coarsen EB Robin 到 coarse levels，避免混合 D/N/R 分类被平均模糊。

验收标准：

- 三个 level 的 EB Robin arrays shape 分别匹配该 level cell domain。
- 每层 cut-cell 上 `abs(a)+abs(b)>0`。
- `lob==rmin` 消失 segments 在所有 levels 一致处理。

### Phase 3.4：AMR HallRZ E-field writeback

目标：

- 每个 level 的 `phi` 写回该 level staggered/Yee `Er/Ez`。

规则：

- 沿用 Stage 1 的 covered/非计算区域保护。
- 不在 covered side 或非计算区域产生有效 `Er/Ez`。
- coarse/fine overlap 的 field 使用 WarpX 现有 MR 语义，不手写破坏 MultiFab ownership。

验收标准：

- `Er/Ez` 无 NaN/Inf。
- covered 区域保护抽查通过。
- 粒子 gather 不读到 EB covered side 的非物理大场。

### Phase 3.5：Stage 3 完整 PIC 验收

必须运行：

- 单层 Stage 1 benchmark 1000 步，确认未回退。
- 三层 AMR HallRZ PIC benchmark 约 1000 步。
- 至少一个 EB Robin 三层 AMR 约 1000 步。
- GPU 与 CPU 多 MPI 都必须覆盖。CPU 总 core 数不超过 4，MPI rank 数不超过 4。

正确性验收：

- 无 NaN/Inf。
- `rho` 不被 solve 修改。
- coarse/fine artificial Dirichlet boundary 连续性达标。
- EB flux/RHS diagnostic 每层达标。
- 粒子数、电荷、壁面损失统计无非物理跳变。
- 与单层高分辨率或阶段性解析/MMS 对照相比，关键物理量误差在预设容差内。

速度验收：

- 单层路径速度不比 Stage 1 慢超过 10%。
- 三层 AMR 相比等效全 finest 单层应有合理加速；若没有，必须解释开销来源。
- Poisson solve time 和 MLMG iter 随 level 无异常爆炸。

Stage 3 完成门槛：

- 所有 Phase 3.1-3.5 通过。
- 用户审查完整 PIC 和速度报告。

## 7. 每次代码修改后的固定检查

每次修改后至少运行：

```bash
bash -lc 'source /home/shuliu/work/gpu_warpx2026/bin/activate && cd /home/shuliu/work/cuda_warpx_202604/WarpX && rg -n "SHULIU-CODEX-DIAG" <modified paths>'
```

```bash
bash -lc 'source /home/shuliu/work/gpu_warpx2026/bin/activate && cd /home/shuliu/work/cuda_warpx_202604/WarpX && git diff --check'
```

```bash
bash -lc 'source /home/shuliu/work/gpu_warpx2026/bin/activate && cd /home/shuliu/work/cuda_warpx_202604/WarpX && git status --short'
```

代码类改动完成后还需：

- 增量 build。
- 对应 phase smoke。
- stage 完成前 1000 步 PIC 完整测试。

## 8. 报告模板

每个 phase 完成报告：

```text
Phase:
Target:
Files changed:
Commands run:
Smoke/tests:
Correctness metrics:
Speed metrics:
Diagnostics remaining:
Risks:
Gate decision:
```

每个 stage 完成报告：

```text
Stage:
PIC benchmark command:
Steps:
MPI/GPU:
Correctness pass/fail:
Speed vs baseline:
Known risks:
User approval needed before next stage:
```

## 9. 执行记录

### 2026-06-05 Stage 0 GPU baseline

已新增并验证：

```text
Test/HallRZ/picmi_hall_rz_pic_benchmark.py
Test/HallRZ/README_HallRZ_EB_Robin_Refactor_Benchmark.md
```

GPU tree 1000 步 baseline 已完成：

```text
command:
python3 Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
  --max-steps 1000 \
  --chunk-steps 50 \
  --summary-json Test/HallRZ/hallrz_stage0_gpu_baseline.json \
  > Test/HallRZ/hallrz_stage0_gpu_baseline.log 2>&1

MPI/GPU: np=1, 1 GPU
steps: 1000
checks.pass: true
avg_step_wall_time_s: 0.1938262595169872
total_step_wall_time_s: 193.8262595169872
particles: electrons=1800, protons=1800
field finite checks: rho_fp=true, phi_fp=true, Er_fp=true, Ez_fp=true
final HallRZ diagnostics: N_NaN=0, N_Inf=0, N_particle_NaN=0, N_particle_Inf=0
warnings: no recorded warnings
```

CPU 多 MPI baseline 在初始检查时尚未完成。当时 CPU tree 只有原始
`Test/HallRZ/picmi_hall_rz_core_smoke.py`，没有本次新增的 benchmark driver。
该 blocker 已在下方 `Stage 0 CPU np4 gate resolved` 记录中解除。

已尝试的 CPU 相关检查：

```text
1. CPU tree 原始 core smoke, np=4, max_steps=1:
   HallRZ step=0 数值诊断正常，但因默认只有 2 个 cell boxes、4 个 MPI ranks
   触发 WarpX high-priority performance warning 并 abort。

2. CPU env 直接运行 GPU tree benchmark, np=4, max_steps=1, warpx_max_grid_size=32:
   失败于初始化前的 Python/PICMI keyword mismatch。CPU Python/PICMI 路径尚不接受
   GPU tree benchmark 使用的 HallRZ solver keyword。这不是 HallRZ 数值失败。
```

为后续 CPU `np=4` benchmark 已在 GPU tree 测试入口增加
`--warpx-max-grid-size` 参数，默认仍为 70；CPU `np=4` 推荐使用 32，使 MPI ranks
有足够 cell boxes。该参数只改变 WarpX box decomposition，不改变物理区域、物理边界、
EB 边界或 HallRZ 几何。

### 2026-06-05 Stage 0 CPU np4 gate resolved

CPU tree 已新增同语义 benchmark driver，但保留 CPU tree 现有
`pywarpx.warpx.hall_rz_*` ParmParse 参数注入方式，不使用 GPU tree 当前的
`ElectrostaticSolver(warpx_hall_rz_*)` keyword 路径。

CPU `np=4` smoke 已通过：

```text
command:
OMP_NUM_THREADS=1 mpirun -np 4 -bind-to none python3 \
  Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
  --max-steps 1 \
  --chunk-steps 1 \
  --summary-json Test/HallRZ/hallrz_stage0_cpu_np4_smoke.json \
  > Test/HallRZ/hallrz_stage0_cpu_np4_smoke.log 2>&1

checks.pass: true
warnings: no recorded warnings
```

CPU `np=4` 1000 步 baseline 已完成：

```text
command:
OMP_NUM_THREADS=1 mpirun -np 4 -bind-to none python3 \
  Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
  --max-steps 1000 \
  --chunk-steps 50 \
  --summary-json Test/HallRZ/hallrz_stage0_cpu_np4_baseline.json \
  > Test/HallRZ/hallrz_stage0_cpu_np4_baseline.log 2>&1

MPI/CPU: np=4, OMP_NUM_THREADS=1
steps: 1000
checks.pass: true
avg_step_wall_time_s: 0.10377750101601123
total_step_wall_time_s: 103.77750101601123
particles: electrons=1800, protons=1800
field finite checks: rho_fp=true, phi_fp=true, Er_fp=true, Ez_fp=true
final HallRZ diagnostics: N_NaN=0, N_Inf=0, N_particle_NaN=0, N_particle_Inf=0
warnings: no recorded warnings
```

原先 CPU `np=4` blocker 已解除：

```text
1. idle-rank warning:
   CPU benchmark 默认 warpx_max_grid_size=32，使 70 x 80 网格在 np=4 下有足够 cell boxes。

2. PICMI keyword mismatch:
   CPU benchmark 不使用 GPU tree 的 HallRZ solver keyword；它沿用 CPU tree 当前
   pywarpx.warpx.hall_rz_* 参数路径。
```

因此 Stage 0 的 GPU baseline gate 和 CPU `np<=4` baseline gate 均已通过。
用户已批准进入 Stage 1。

### 2026-06-05 Stage 1 GPU tree execution

已完成的 Stage 1 修改：

```text
WarpX HallRZ solver:
- 使用 linop.setEBNodalGeometry(0, 0, ...) 只上传 finest-level planar nodal geometry。
- Neumann mode 使用 linop.setEBInhomogNeumann(0, eb_gn_cc)。
- Robin mode 使用 linop.setEBFVMRobin(0, eb_a_cc, eb_b_cc, eb_f_cc)。
- 删除 HallRZ 三整数 MG stop rule 使用路径。
- 停用旧 RZ metric/weighted compatibility setters。

Python/API:
- 新增 warpx.hall_rz_eb_bc_mode = neumann|robin。
- 新增 hallrz.set_eb_robin(a,b,f,lev=0)。
- 新增 hallrz.clear_eb_robin(lev=0)。
- 新增 hallrz.has_eb_robin(lev=0)。

lob == rmin:
- EB2 不构造 low covered block。
- HallRZ segment 0/2 不存在。
- Neumann/Robin 数据填充不消费消失 segment。
```

已通过的 Stage 1 GPU 正确性测试：

```text
Build:
cmake --build build -j 8 --target pip_install_nodeps

Short smoke:
- Neumann 2 steps
- Neumann-equivalent Robin 2 steps
- rmin=0,lob=0 2 steps
- rmin>0,lob=rmin with physical lo-r Dirichlet 2 steps
```

1000-step GPU gates：

```text
stage1_neumann_baseline_equiv:
  summary: Test/HallRZ/hallrz_stage1_gpu_neumann_baseline_equiv.json
  log:     Test/HallRZ/hallrz_stage1_gpu_neumann_baseline_equiv.log
  checks.pass: true
  avg_step_wall_time_s: 0.23310955688001558
  particles: electrons=1800, protons=1800
  field finite checks: true

stage1_robin_neumann_equiv:
  summary: Test/HallRZ/hallrz_stage1_gpu_robin_neumann_equiv.json
  log:     Test/HallRZ/hallrz_stage1_gpu_robin_neumann_equiv.log
  checks.pass: true
  avg_step_wall_time_s: 0.23312457087097574
  particles: electrons=1800, protons=1800
  field finite checks: true
  Neumann-vs-Robin field L2 differences: ~1e-7 level

stage1_lob_eq_rmin_axis:
  summary: Test/HallRZ/hallrz_stage1_gpu_lob_eq_rmin_axis.json
  log:     Test/HallRZ/hallrz_stage1_gpu_lob_eq_rmin_axis.log
  checks.pass: true
  avg_step_wall_time_s: 0.13331105433301127
  particles: electrons=1800, protons=1800
  field finite checks: true
  EB total RZ flux: -0.0032
```

Stage 1 当前 blocker：

```text
Stage 0 GPU baseline avg_step_wall_time_s:
  0.1938262595169872

Stage 1 Neumann baseline-equivalent avg_step_wall_time_s:
  0.23310955688001558
  delta vs Stage 0: +20.27%

Stage 1 Neumann-equivalent Robin avg_step_wall_time_s:
  0.23312457087097574
  delta vs Stage 0: +20.28%
```

由于原圆环路径速度退化超过本计划规定的 `10%` 门槛，Stage 1 不能标记完成，
也不允许进入 Stage 2。

初步 profiler 证据：

```text
Stage 0 baseline:
  MLEBNodeFVLaplacian::Fapply() total time ~32.93 s

Stage 1 original annulus Neumann:
  MLEBNodeFVLaplacian::Fapply() total time ~43.9 s

Stage 1 Robin:
  Robin 本身没有可见额外开销；Neumann 与 Robin 总步时几乎相同。

Stage 1 prepareForSolve:
  总时间低于 1 s，不是主退化源。
```

后续继续施工前必须先处理或重新评审该速度 gate：

```text
1. 若认为 AMReX 自动 MG EB-FVM records 的额外 operator work 是正确性必须付出的代价，
   需用户确认是否放宽 Stage 1 速度门槛。
2. 若不放宽，下一步应新增性能定位小 phase：
   - 对比 old manual MG geometry 与 AMReX auto MG geometry 的 operator work；
   - 定位 Fapply/Fsmooth 调用数和每次调用成本增加原因；
   - 不得恢复旧 RZ metric/weighted switches；
   - 不得恢复 HallRZ 三整数 MG stop rule。
3. CPU np=4 Stage 1 gate 尚未运行；在 GPU speed blocker 解除或用户确认放宽前，
   不能把 Stage 1 视为完成。
```

### 2026-06-06 Stage 1 MG/MCL 性能恢复计划补充

AMReX 端已完成 EB-FVM changing-BC correctness fix、几何/BC flux record 缓存拆分和
P3B open-face operator cache。当前 WarpX 侧不继续修改 AMReX；若后续发现 AMReX generic
EB-FVM 问题，只能先报告证据并经用户确认后另行处理。

本次计划修订结论：

```text
1. WarpX C++ 不做 HallRZ MCL 自动性能启发式。
2. Python/ParmParse 负责指定 warpx.hall_rz_max_coarsening_level。
3. WarpX C++ 只把该值作为 AMReX LPInfo::setMaxCoarseningLevel upper bound 传入。
4. AMReX 继续决定实际 MG levels。
5. Stage 1 先补 MCL 性能矩阵和 requested/actual NMG 诊断，再判断速度 gate。
6. 每个 PIC 时间步 Poisson solve 必须使用上一 PIC 时间步 phi 热启动，不能每步冷启动。
```

已补入计划正文：

```text
Section 2:
  增加 HallRZ Poisson solve 热启动全局约束。

Phase 1.5:
  新增 Stage 1 MG/MCL 性能恢复补充 gate。
  要求 benchmark 增加 --hall-rz-max-coarsening-level。
  要求日志记录 requested_mcl 和 actual_nmg_levels。
  要求跑 MCL=0/1/2/default 矩阵。
  要求热启动正确性验收，防止 stale solution 或 0-iter 假快。

Phase 1.6:
  原 Stage 1 完整 PIC gate 后移，并要求使用 Phase 1.5 选定显式 MCL。

Stage 2/3:
  增加 per-AMR-level MCL 向量设计：
  warpx.hall_rz_max_coarsening_levels = M0 M1 M2
  向量顺序是 AMR level 0/1/2。
```

当前 Stage 1 仍未完成。下一步正式施工前，应先实现 Phase 1.5 的 benchmark 参数、C++ 诊断和
热启动修正，然后跑 GPU MCL 矩阵。矩阵通过后再运行 CPU `np<=4` gate 和 Stage 1 完整 PIC gate。

### 2026-06-06 Stage 1.5 GPU MCL 矩阵执行记录

本轮基于 AMReX CUDA build blocker 修复后的冻结版本继续测试 WarpX 端。已完成：

```text
WarpX build:
  cmake --build build -j 8 --target pip_install_nodeps
  result: pass

2-step smoke:
  Neumann MCL=0: checks.pass=true, requested_mcl=0, actual_nmg_levels=1
  Neumann MCL=1: checks.pass=true, requested_mcl=1, actual_nmg_levels=2
  Robin   MCL=0: checks.pass=true, requested_mcl=0, actual_nmg_levels=1
```

1000-step GPU MCL matrix:

```text
Stage 0 old GPU baseline:
  Test/HallRZ/hallrz_stage0_gpu_baseline.json
  avg_step_wall_time_s = 0.193826259517

Neumann MCL=0:
  summary = Test/HallRZ/hallrz_stage1_gpu_neumann_mcl0_matrix.json
  log     = Test/HallRZ/hallrz_stage1_gpu_neumann_mcl0_matrix.log
  checks.pass = true
  avg_step_wall_time_s = 0.029709197016
  actual_nmg_levels = 1

Neumann MCL=1:
  summary = Test/HallRZ/hallrz_stage1_gpu_neumann_mcl1_matrix.json
  log     = Test/HallRZ/hallrz_stage1_gpu_neumann_mcl1_matrix.log
  checks.pass = true
  avg_step_wall_time_s = 0.037353753054
  actual_nmg_levels = 2

Neumann MCL=2:
  summary = Test/HallRZ/hallrz_stage1_gpu_neumann_mcl2_matrix.json
  log     = Test/HallRZ/hallrz_stage1_gpu_neumann_mcl2_matrix.log
  checks.pass = true
  avg_step_wall_time_s = 0.037426452689
  actual_nmg_levels = 2

Neumann MCL=30:
  summary = Test/HallRZ/hallrz_stage1_gpu_neumann_mcl30_matrix.json
  log     = Test/HallRZ/hallrz_stage1_gpu_neumann_mcl30_matrix.log
  checks.pass = true
  avg_step_wall_time_s = 0.036343555479
  actual_nmg_levels = 2

Robin MCL=0:
  summary = Test/HallRZ/hallrz_stage1_gpu_robin_mcl0_matrix.json
  log     = Test/HallRZ/hallrz_stage1_gpu_robin_mcl0_matrix.log
  checks.pass = true
  avg_step_wall_time_s = 0.030235233473
  actual_nmg_levels = 1

Robin MCL=30:
  summary = Test/HallRZ/hallrz_stage1_gpu_robin_mcl30_matrix.json
  log     = Test/HallRZ/hallrz_stage1_gpu_robin_mcl30_matrix.log
  checks.pass = true
  avg_step_wall_time_s = 0.036079597294
  actual_nmg_levels = 2
```

相对 Neumann MCL=0 的速度：

```text
Neumann MCL=1:  +25.73%
Neumann MCL=2:  +25.98%
Neumann MCL=30: +22.33%
Robin   MCL=0:   +1.77%
Robin   MCL=30: +21.44%
```

正确性与热启动观测：

```text
All matrix cases:
  fields_finite = true
  particles_positive = true
  N_NaN = 0
  N_Inf = 0
  particle NaN/Inf/out-of-domain checks pass in HallRZ summary

Poisson summary count:
  21 summaries per 1000-step run, covering step 0 ... 1000

Final Iter:
  MCL=0 Neumann/Robin: min/max = 0/6, mean ~1.19
  MCL>=1 Neumann:      min/max = 0/8, mean ~1.29
  Robin MCL=30:        min/max = 0/9, mean ~1.48

resid/resid0:
  max observed in summaries is O(1e-10), no stale residual evidence from logs.
```

Neumann-equivalent Robin 对照：

```text
MCL=0 final field differences:
  phi_fp dmax ~7.3e-09
  Er_fp  dmax ~6.3e-08
  Ez_fp  dmax ~1.1e-07
  rho_fp dmax ~2.1e-19

MCL=30 final field differences:
  phi_fp dmax ~2.0e-10
  Er_fp  dmax ~5.0e-08
  Ez_fp  dmax ~9.3e-09
  rho_fp dmax ~9.2e-18
```

当前 Phase 1.5 判断：

```text
1. Python benchmark 参数 --hall-rz-max-coarsening-level 已能传到 C++ 和 AMReX。
2. C++ 日志已能显示 requested_mcl 和 actual_nmg_levels。
3. WarpX HallRZ 单层 solve 已从每步冷启动改为使用上一 PIC step 的 phi 热启动。
4. 对当前 70x80 benchmark，MCL=0 是最快推荐显式设置。
5. MCL=1/2/30 都实际保留 2 个 MG levels，速度慢于 MCL=0 约 22-26%。
6. Robin 与 Neumann 在同 MCL 下速度和场量等价；无需补 Robin MCL=1/2。
7. 相对旧 Stage 0 baseline，本轮 AMReX fix + WarpX 热启动后的所有矩阵点都明显更快。
```

Stage 状态：

```text
Phase 1.5 GPU MCL matrix: pass, 推荐 Stage 1 后续 gate 使用显式 MCL=0。
Stage 1 尚未完成：
  CPU np<=4 gate 尚未在本轮 AMReX fix 后重跑。
  Phase 1.6 完整 PIC gate 尚未跑完。
  lob == rmin 的 1000-step gate 仍需按 Phase 1.6 要求复查。
```

### 2026-06-06 Stage 1.6 GPU 完整 PIC gate 补充记录

本轮继续使用 Phase 1.5 推荐显式设置：

```text
warpx.hall_rz_max_coarsening_level = 0
```

已由 Phase 1.5 GPU matrix 覆盖的 1000-step gate：

```text
Baseline-equivalent Neumann:
  summary = Test/HallRZ/hallrz_stage1_gpu_neumann_mcl0_matrix.json
  log     = Test/HallRZ/hallrz_stage1_gpu_neumann_mcl0_matrix.log
  checks.pass = true
  avg_step_wall_time_s = 0.029709197016
  actual_nmg_levels = 1

Neumann-equivalent Robin:
  summary = Test/HallRZ/hallrz_stage1_gpu_robin_mcl0_matrix.json
  log     = Test/HallRZ/hallrz_stage1_gpu_robin_mcl0_matrix.log
  checks.pass = true
  avg_step_wall_time_s = 0.030235233473
  actual_nmg_levels = 1
```

新增 `lob == rmin` GPU 1000-step gate：

```text
rmin=0, lob=0, axis lo-r:
  command case = stage1_gpu_lob_eq_rmin_axis_mcl0_full
  summary = Test/HallRZ/hallrz_stage1_gpu_lob_eq_rmin_axis_mcl0_full.json
  log     = Test/HallRZ/hallrz_stage1_gpu_lob_eq_rmin_axis_mcl0_full.log
  checks.pass = true
  avg_step_wall_time_s = 0.030990342163
  particles = electrons 1800, protons 1800
  actual_nmg_levels = 1
  final q_EB_total_RZ = -0.0032
  N_NaN/N_Inf = 0/0
  particle NaN/Inf/out-of-domain/inside-covered = 0/0/0/0

rmin=0.02, lob=0.02, physical lo-r Dirichlet:
  command case = stage1_gpu_lob_eq_rmin_rlo_dirichlet_mcl0_full
  summary = Test/HallRZ/hallrz_stage1_gpu_lob_eq_rmin_rlo_dirichlet_mcl0_full.json
  log     = Test/HallRZ/hallrz_stage1_gpu_lob_eq_rmin_rlo_dirichlet_mcl0_full.log
  checks.pass = true
  avg_step_wall_time_s = 0.025553746462
  particles = electrons 1680, protons 1680
  actual_nmg_levels = 1
  final q_EB_total_RZ = -0.0032
  N_NaN/N_Inf = 0/0
  particle NaN/Inf/out-of-domain/inside-covered = 0/0/0/0
```

MPI/CPU gate 状态：

```text
CPU backend probe:
  env = /home/shuliu/work/cpu_warpx2026/bin/activate
  command case = stage1_cpu_probe
  result = blocked before WarpX initialize
  reason = CPU Python/PICMI binding does not recognize current HallRZ keywords:
           warpx_hall_rz_enable, warpx_hall_rz_lob, warpx_hall_rz_eb_bc_mode, ...
  interpretation = CPU backend gate is not failed numerically; the CPU binding/build is stale for
                   this GPU-tree HallRZ interface.

GPU-env MPI fallback:
  command case = stage1_gpu_mpi_np4_mcl0_50step
  summary = Test/HallRZ/hallrz_stage1_gpu_mpi_np4_mcl0_50step.json
  log     = Test/HallRZ/hallrz_stage1_gpu_mpi_np4_mcl0_50step.log
  MPI = np=4, OMP_NUM_THREADS=1
  hardware = 4 ranks sharing 1 CUDA GPU; correctness-only, not speed baseline
  steps = 50
  checks.pass = true
  avg_step_wall_time_s = 0.362558866280
  actual_nmg_levels = 1
  N_NaN/N_Inf = 0/0
  particle NaN/Inf/out-of-domain/inside-covered = 0/0/0/0
  warnings = no recorded warnings
```

速度对照：

```text
Old Stage 0 unmodified GPU baseline:
  avg_step_wall_time_s = 0.193826259517

Earlier Stage 1 pre-performance-fix Neumann:
  avg_step_wall_time_s = 0.233109556880

Earlier Stage 1 pre-performance-fix Robin:
  avg_step_wall_time_s = 0.233124570871

Current MCL=0 Neumann:
  avg_step_wall_time_s = 0.029709197016
  vs old Stage 0 baseline: 0.153277x, -84.67%, 6.53x faster
  vs earlier Stage 1 Neumann: 0.127448x, -87.26%, 7.85x faster

Current MCL=0 Robin:
  avg_step_wall_time_s = 0.030235233473
  vs old Stage 0 baseline: 0.155991x, -84.40%, 6.41x faster
  vs earlier Stage 1 Robin: 0.129697x, -87.03%, 7.71x faster
```

Stage 1 状态：

```text
GPU Phase 1.6 required 1000-step gates: pass.
np=4 MPI decomposition fallback: pass for 50 steps on shared GPU.
True CPU backend np<=4 1000-step gate: blocked by stale CPU Python/PICMI binding; deferred.

User decision on 2026-06-06:
  If the np=4 shared-GPU MPI correctness test is correct, Stage 1 may count the parallel gate as pass.
  Future tests should first use np=4 shared-GPU MPI for the multi-MPI no-error purpose.

Stage 1 gate decision:
  pass for current GPU-tree Stage 1 scope.
  CPU backend gate remains a deferred follow-up, not a blocker for entering Stage 2 discussion.
```

### 2026-06-06 Stage 2 AMR 初始化框架执行记录

本轮 Stage 2 只实现和验证 HallRZ 静态三层 AMR grid initialization/tagging 框架，不进入
HallRZ 多层 Poisson solve。

已完成的 Stage 2 修改：

```text
WarpX C++:
  Source/WarpX.H
  Source/WarpX.cpp

  - 新增 warpx.hall_rz_amr_enable。
  - 新增 warpx.hall_rz_amr_r1/z1/r2/z2。
  - 新增 warpx.hall_rz_max_coarsening_levels = M0 M1 M2 解析与校验。
  - HallRZ AMR 模式要求:
      amr.max_level = 2
      amr.ref_ratio = 2 2
      warpx.regrid_int = -1
      rmin <= lob < hib < r2 < r1 < rmax
      zmin < out < z2 < z1 < zmax
      r1/z1 落在 level-0 faces
      r2/z2 落在 level-1 faces
      lob/hib/out 落在 level-2 faces
  - HallRZ AMR 模式禁止同时设置 warpx.fine_tag_lo/hi 或 ref_patch_function(x,y,z)。
  - 非 AMR HallRZ 仍要求 amr.max_level=0，避免静默进入未实现的多层 solve。
  - WarpX::ErrorEst 新增 HallRZ level-aware tag rule:
      lev=0 tags Ω1
      lev=1 tags Ω2
      lev>=2 no tags

Python/test:
  Python/pywarpx/picmi.py
  Test/HallRZ/picmi_hall_rz_pic_benchmark.py

  - PICMI HallRZ solver kwargs 允许透传 amr_enable/amr_r1/amr_z1/amr_r2/amr_z2。
  - benchmark 新增 --init-only、--hall-rz-amr-enable、
    --hall-rz-amr-r1/z1/r2/z2、--hall-rz-max-coarsening-levels。
  - AMR init-only grid validation 使用 warpx.do_electrostatic=none，避免触发当前单层
    HallRZ Poisson solve 限制；该模式只验证 grid/tagging/parameter contract，不做 Poisson solve。
  - AMR init-only summary 记录 max_level、finest_level、每层 box 数和 minimal box。
```

Build：

```text
cmake --build build -j 8 --target pip_install_nodeps
result: pass
```

Stage 2 单层生产路径回归：

```text
command case = stage2_single_level_neumann_mcl0_full
summary = Test/HallRZ/hallrz_stage2_single_level_neumann_mcl0_full.json
log     = Test/HallRZ/hallrz_stage2_single_level_neumann_mcl0_full.log

steps = 1000
MPI/GPU = np=1, 1 GPU
checks.pass = true
avg_step_wall_time_s = 0.029417214385
total_step_wall_time_s = 29.417214385
amr.max_level/finest_level = 0/0
```

速度对照：

```text
Old Stage 0 unmodified GPU baseline:
  avg_step_wall_time_s = 0.193826259517

Stage 1 recommended MCL=0 Neumann:
  avg_step_wall_time_s = 0.029709197016

Stage 2 single-level Neumann MCL=0:
  avg_step_wall_time_s = 0.029417214385
  vs old Stage 0 baseline: 0.15177x, -84.82%, 6.59x faster
  vs Stage 1 MCL=0:        0.99017x,  -0.98%, effectively unchanged/slightly faster
```

Stage 2 AMR grid initialization smoke：

```text
command case = stage2_amr_init_only_smoke
summary = Test/HallRZ/hallrz_stage2_amr_init_only_smoke.json
log     = Test/HallRZ/hallrz_stage2_amr_init_only_smoke.log

MPI/GPU = np=1, 1 GPU
checks.pass = true
max_level/finest_level = 2/2
warpx_max_grid_size = 256
requested AMR:
  r1=0.060, z1=0.060
  r2=0.055, z2=0.050
generated boxes:
  level 0: num_boxes=1, minimal_box length=(70,80), small_end=(0,0)
  level 1: num_boxes=1, minimal_box length=(122,122), small_end=(0,0)
  level 2: num_boxes=1, minimal_box length=(222,202), small_end=(0,0)

interpretation:
  level 1/2 boxes cover Ω1/Ω2 and are lower-left anchored.
  They are larger than the exact requested rectangles by AMReX proper nesting / grid buffer,
  which is acceptable for Stage 2 because the gate requires coverage, not exact equality.
```

Stage 2 AMR init-only 多 MPI gate：

```text
command case = stage2_amr_init_only_mpi_np4_shared_gpu
summary = Test/HallRZ/hallrz_stage2_amr_init_only_mpi_np4_shared_gpu.json
log     = Test/HallRZ/hallrz_stage2_amr_init_only_mpi_np4_shared_gpu.log

MPI/GPU = np=4 sharing 1 GPU, correctness-only
checks.pass = true
max_level/finest_level = 2/2
generated boxes:
  level 0: num_boxes=2, minimal_box length=(70,80), small_end=(0,0)
  level 1: num_boxes=4, minimal_box length=(122,122), small_end=(0,0)
  level 2: num_boxes=12, minimal_box length=(222,202), small_end=(0,0)
```

Stage 2 单层多 MPI smoke：

```text
command case = stage2_single_level_mpi_np4_mcl0_50step
summary = Test/HallRZ/hallrz_stage2_single_level_mpi_np4_mcl0_50step.json
log     = Test/HallRZ/hallrz_stage2_single_level_mpi_np4_mcl0_50step.log

MPI/GPU = np=4 sharing 1 GPU, correctness-only
steps = 50
checks.pass = true
avg_step_wall_time_s = 0.349977493180
max_level/finest_level = 0/0
```

非法输入负例：

```text
command case = stage2_amr_bad_alignment_negative
log = Test/HallRZ/hallrz_stage2_amr_bad_alignment_negative.log
input: hall_rz_amr_r2 = 0.0551
result: expected abort
message includes:
  HallRZ AMR requires r1/z1 on level-0 faces, r2/z2 on level-1 faces,
  and hall_rz_lob/hall_rz_hib/hall_rz_out on level-2 faces.
```

当前 Stage 2 判断：

```text
Phase 2.1 parameter interface/validation: pass.
Phase 2.2 level-aware tagging: pass.
Phase 2.3 AMR geometry validation: pass.
Phase 2.4 current GPU-tree gate:
  single-level 1000-step PIC regression: pass.
  AMR grid init-only np=1: pass.
  AMR grid init-only np=4 shared GPU: pass.
  single-level np=4 shared GPU smoke: pass.

Stage 2 gate decision:
  pass for current GPU-tree initialization-framework scope.
  This does not implement or validate HallRZ multi-level Poisson solve.
  Entering Stage 3 still requires user review/approval of this record.
```

### 2026-06-07 Stage 3.1 driver skeleton 执行记录

本轮开始 Stage 3，只实现 Phase 3.1 driver skeleton，不实现多层 Poisson 数值求解。

已完成的 Stage 3.1 修改：

```text
Source/FieldSolver/ElectrostaticSolvers/LabFrameExplicitES.cpp

HallRZ max_level==0:
  保持原单层 ComputePhiAndE 路径不变。

HallRZ max_level==2:
  在 rho deposition 和 SyncRho 后进入 HallRZ AMR driver skeleton。
  skeleton 打印：
    step
    solve_order=0->1->2
    每个 AMR level 的 requested_mcl
    rho_boxes / phi_boxes
    cell_valid_region
    physical domain
    rlo/rhi/zlo/zhi face role
    EB boundary role
    internal box faces 仅为 decomposition，不是 BC
  然后明确 abort，说明以下内容尚未实现：
    level-by-level HallRZ Poisson solve
    coarse Dirichlet artificial boundaries
    per-level EB data
    AMR E-field writeback
```

计划修订/澄清：

```text
Phase 3.1 原计划要求 driver 日志打印 actual_nmg_levels。
施工中确认：在 skeleton 阶段尚未构造每层 MLEBNodeFVLaplacian/MLMG，因此没有真实
actual NMG levels。当前 skeleton 日志写：

  actual_nmg_levels=not_constructed_in_phase_3_1

真实 actual_nmg_levels 验收移到后续第一次真正构造 per-level linop 的 phase；届时必须打印
每个 AMR level 的 requested MCL 和实际 linop.NMGLevels(0)。
```

Build：

```text
cmake --build build -j 8 --target pip_install_nodeps
result: pass
```

单层回归 smoke：

```text
command case = stage3_1_single_level_regression_smoke
summary = Test/HallRZ/hallrz_stage3_1_single_level_regression_smoke.json
log     = Test/HallRZ/hallrz_stage3_1_single_level_regression_smoke.log

steps = 2
MPI/GPU = np=1, 1 GPU
checks.pass = true
max_level/finest_level = 0/0
avg_step_wall_time_s = 0.002269986508
```

AMR driver skeleton expected-abort：

```text
command case = stage3_1_amr_driver_skeleton_expected_abort
log = Test/HallRZ/hallrz_stage3_1_amr_driver_skeleton_expected_abort.log

input:
  max_level=2
  hall_rz_amr_r1/z1 = 0.060 / 0.060
  hall_rz_amr_r2/z2 = 0.055 / 0.050
  hall_rz_max_coarsening_levels = 0 1 2
  amrex.the_arena_init_size = 0  # short smoke only, avoids GPU default arena OOM

expected result:
  abort after Stage 3.1 skeleton logs, not at old single-level assertion.

observed skeleton log:
  level 0:
    requested_mcl=0
    rlo/rhi/zlo/zhi = physical domain faces
  level 1:
    requested_mcl=1
    rlo/zlo = physical domain lo faces
    rhi/zhi = amr-artificial-hi-coarse-dirichlet-pending
  level 2:
    requested_mcl=2
    rlo/zlo = physical domain lo faces
    rhi/zhi = amr-artificial-hi-coarse-dirichlet-pending
  all levels:
    eb_boundary = HallRZ-EB-FVM-fluid-to-covered
    internal_box_faces = decomposition-only-not-BC

abort message:
  HallRZ AMR driver skeleton reached. Level-by-level HallRZ Poisson solve,
  coarse Dirichlet artificial boundaries, per-level EB data, and AMR E-field
  writeback are planned for later Stage 3 phases and are not implemented yet.
```

当前 Stage 3.1 判断：

```text
Phase 3.1 driver skeleton: pass for current GPU-tree scope.
max_level==0 path: smoke pass.
max_level==2 path: enters explicit skeleton and aborts with correct boundary-role logs.
Stage 3.2 is not started yet.
```

### 2026-06-07 Stage 3.2 fine patch artificial boundary 执行记录

本轮开始 Phase 3.2，只在 WarpX 端实现 HallRZ AMR level-by-level solve 的 coarse-to-fine
warm-start / artificial Dirichlet 数据流。AMReX 只做源码/API contract 调查，未修改。

接口证据：

```text
AMReX_MLLinOp.H:
  setCoarseFineBC() 注释明确写 "For cell-centered solves only"。

AMReX_MLNodeLaplacian.cpp:
  checkpoint 注释写 "m_coarse_data_for_bc: not used"。

AMReX Tests/LinearSolvers/NodalPoisson:
  nodal level-by-level solve 不调用 setCoarseFineBC；
  fine level solve 前用 InterpFromCoarseLevel 把 coarse solution 插值进 fine solution；
  注释明确该插值用于 coarse/fine Dirichlet boundary。

AMReX_MLNodeLinOp:
  buildMasks() 用 cell-centered grid coverage 生成 nodal m_dirichlet_mask。
  BuildMask(ccdomain, period, 0, 1, 2, 0) 后，
    valid cells/interior = 0；
    same-level valid region 外但 physical domain 内 = 1；
    physical domain 外 = 2。
  mlndlap_set_dirichlet_mask() 会把 adjacent cell mask == 1 的 patch-boundary nodes
  标成 Dirichlet；内部 box/FAB/MPI 分块面因为同一 level coverage 是 covered=0，
  不会被标成人工 Dirichlet。

AMReX_MLNodeLinOp / MLEBNodeFVLaplacian:
  solutionResidual() 在 Dirichlet mask nodes 上 residual=0。
  Fapply/Fsmooth EB-FVM kernels 接收 dmask，Dirichlet nodes 不作为未知量更新。
```

由此确认：HallRZ nodal EB-FVM AMR 首版不应调用 `setCoarseFineBC()` 作为人工边界主机制；
正确 WarpX 侧机制是：

```text
solve level lev
  -> InterpolateHallRZPhiBetweenLevels(phi[lev], phi[lev+1])
       temporary coarse MultiFab uses coarsened fine BoxArray + fine DistributionMap
       data transfer uses ablastr::utils::communication::ParallelCopy
       local interpolation uses ablastr::fields::details::PoissonInterpCPtoFP
  -> fine phi now contains coarse interpolation on artificial boundary nodes
  -> next level MLEBNodeFVLaplacian buildMasks marks patch-boundary nodes as Dirichlet
  -> MLMG solve uses interpolated phi as both known artificial boundary values and warm-start guess
```

已完成的 Stage 3.2 修改：

```text
Source/FieldSolver/ElectrostaticSolvers/HallRZPoissonSolver.H
Source/FieldSolver/ElectrostaticSolvers/HallRZPoissonSolver.cpp
  ComputePhiAndE() 增加可选参数：
    max_coarsening_level_override
    amr_level
  单层默认调用保持兼容。
  solve 日志现在打印 amr_level、requested_mcl、actual_nmg_levels。

Source/FieldSolver/ElectrostaticSolvers/LabFrameExplicitES.cpp
  新增 InterpolateHallRZPhiBetweenLevels()。
  max_level==2 HallRZ path 从 Stage 3.1 expected-abort 改为 level-by-level loop：
    lev=0 -> solve -> interpolate phi0 to phi1
    lev=1 -> solve -> interpolate phi1 to phi2
    lev=2 -> solve
  每层使用当前 rho/phi/E MultiFab、warpx.Geom(lev)、phi DistributionMap、
  warpx.fieldEBFactory(lev) 和 per-level requested MCL。
```

当前 Stage 3.2 明确限制：

```text
三层 AMR level-by-level solve 当前只允许 scalar boundary inputs。

禁止在 Stage 3.2 使用单层 Python-provided boundary arrays：
  hallrz.set_eb_neumann(...)
  hallrz.set_eb_robin(...)
  physical z-hi/r-hi/r-lo/inlet Python arrays

原因：
  这些数组当前是单层 global shape，不能安全用于 level 1/2 不同 domain / valid region。
  per-level Python EB/physical boundary data 属于后续 Stage 3.3/边界数据 phase。

当前代码在 max_level==2 且上述 Python arrays active 时明确 abort：
  "Stage-3.2 HallRZ AMR level-by-level solve currently requires scalar boundary inputs..."
```

Build：

```text
cmake --build build -j 8 --target pip_install_nodeps
result: pass
```

单层回归 smoke：

```text
command case = stage3_2_single_level_post_guard_smoke
summary = Test/HallRZ/hallrz_stage3_2_single_level_post_guard_smoke.json
log     = Test/HallRZ/hallrz_stage3_2_single_level_post_guard_smoke.log

steps = 1
MPI/GPU = np=1, 1 GPU
checks.pass = true
avg_step_wall_time_s = 0.002642242005
HallRZ Poisson:
  amr_level=0
  requested_mcl=0
  actual_nmg_levels=1
  Final Iter=6
  resid/resid0=5.158500149e-11
  N_NaN=0, N_Inf=0
```

三层 AMR scalar-boundary smoke：

```text
command case = stage3_2_amr_level_solve_scalar_smoke
summary = Test/HallRZ/hallrz_stage3_2_amr_level_solve_scalar_smoke.json
log     = Test/HallRZ/hallrz_stage3_2_amr_level_solve_scalar_smoke.log

input:
  max_level=2
  scalar_boundaries=true
  warpx_max_grid_size=280
  hall_rz_amr_r1/z1 = 0.060 / 0.060
  hall_rz_amr_r2/z2 = 0.055 / 0.050
  hall_rz_max_coarsening_levels = 0 1 2

checks.pass = true
avg_step_wall_time_s = 0.214596531994  # short smoke only, not speed gate

level 0:
  requested_mcl=0
  actual_nmg_levels=1
  Final Iter=6
  resid/resid0=2.735924917e-11
  N_NaN=0, N_Inf=0

level 1:
  requested_mcl=1
  actual_nmg_levels=2
  Final Iter=4
  resid/resid0=3.159655053e-12
  rhi/zhi = amr-artificial-hi-coarse-dirichlet
  N_NaN=0, N_Inf=0

level 2:
  requested_mcl=2
  actual_nmg_levels=2
  Final Iter=4
  resid/resid0=1.770138263e-12
  rhi/zhi = amr-artificial-hi-coarse-dirichlet
  N_NaN=0, N_Inf=0

particle summary:
  Np_total=3600
  N_particle_NaN=0
  N_particle_Inf=0
  N_out_of_domain=0
  N_inside_covered_EB=0
```

三层 AMR scalar-boundary MPI correctness smoke：

```text
command case = stage3_2_amr_level_solve_scalar_mpi_np4_smoke
summary = Test/HallRZ/hallrz_stage3_2_amr_level_solve_scalar_mpi_np4_smoke.json
log     = Test/HallRZ/hallrz_stage3_2_amr_level_solve_scalar_mpi_np4_smoke.log

MPI/GPU = np=4 sharing 1 GPU
purpose = MPI/box decomposition correctness only, not speed baseline
warpx_max_grid_size=140

checks.pass = true
nprocs = 4
level boxes = 4 / 4 / 4
no recorded warnings

level 0:
  requested_mcl=0
  actual_nmg_levels=1
  Final Iter=6
  resid/resid0=3.655770457e-11
  N_NaN=0, N_Inf=0

level 1:
  requested_mcl=1
  actual_nmg_levels=2
  Final Iter=4
  resid/resid0=3.098438292e-12
  N_NaN=0, N_Inf=0

level 2:
  requested_mcl=2
  actual_nmg_levels=2
  Final Iter=4
  resid/resid0=1.408761914e-12
  N_NaN=0, N_Inf=0
```

负向 guard smoke：

```text
command case = stage3_2_amr_python_bc_guard_expected_abort
log = Test/HallRZ/hallrz_stage3_2_amr_python_bc_guard_expected_abort.log

input:
  max_level=2
  Python boundary arrays active by default benchmark path

expected result:
  abort before per-level solve with clear Stage-3.2 scalar-boundary-only message.

observed:
  pass expected-abort; log contains "scalar boundary inputs" message.
```

当前 Stage 3.2 判断：

```text
Phase 3.2 fine patch artificial boundary data path: pass for current scalar-boundary
GPU-tree smoke scope.

已满足：
  - level 0/1/2 按顺序 solve；
  - coarse-to-fine interpolation 发生在下一层 solve 前；
  - requested MCL 和 actual NMG levels 按 AMR level 打印；
  - level 1/2 high faces 识别为 AMR artificial coarse Dirichlet；
  - lo-r/lo-z physical boundary 仍按 physical BC 处理；
  - MPI np=4 shared-GPU correctness smoke pass；
  - 单层路径 smoke pass。

未完成，不能视为 Stage 3 完成：
  - per-level Python EB/physical BC arrays 尚未实现；
  - EB Robin 三层 AMR 完整测试尚未跑；
  - coarse/fine continuity quantitative diagnostic 尚未实现；
  - 1000 步完整 PIC gate 尚未跑；
  - Stage 3.3/3.4/3.5 仍需继续。
```

### 2026-06-07 Stage 3.3 per-level EB boundary data 执行记录

本轮继续 Stage 3.3，只在 WarpX 端实现 HallRZ AMR per-level Python EB Neumann / EB Robin
数据，不修改 AMReX。

已完成的 Stage 3.3 修改：

```text
Source/FieldSolver/ElectrostaticSolvers/HallRZPoissonSolver.H
Source/FieldSolver/ElectrostaticSolvers/HallRZPoissonSolver.cpp
  Python EB Neumann / EB Robin runtime data 从单个全局数组改为最多三层 per-level 存储：
    lev=0,1,2
  set/clear/has API 增加 level 参数：
    SetPythonEBNeumann(..., lev=0)
    ClearPythonEBNeumann(lev=-1)
    HasPythonEBNeumann(lev=-1)
    SetPythonEBRobin(..., lev=0)
    ClearPythonEBRobin(lev=-1)
    HasPythonEBRobin(lev=-1)
  ComputePhiAndE(..., amr_level) 根据当前 AMR level 选择对应 EB data。
  EB Robin 仍校验 active cut-cell 上 finite 且 abs(a)+abs(b)>0。

Source/Python/pyWarpX.cpp
Python/pywarpx/hallrz.py
  hallrz.set_eb_neumann(g, lev=0)
  hallrz.set_eb_robin(a,b,f, lev=0)
  hallrz.clear_eb_neumann(lev=None)   # None clears all levels
  hallrz.clear_eb_robin(lev=None)     # None clears all levels
  hallrz.has_eb_neumann(lev=None)     # None queries any level
  hallrz.has_eb_robin(lev=None)       # None queries any level

Source/FieldSolver/ElectrostaticSolvers/LabFrameExplicitES.cpp
  Stage 3.2 scalar-only guard 替换为 Stage 3.3 boundary-data guard：
    AMR Robin mode 要求 lev 0/1/2 都有 EB Robin data；
    AMR Neumann mode 允许全 scalar，或要求 lev 0/1/2 都有 EB Neumann data；
    AMR path 仍禁止 physical domain Python arrays。

Test/HallRZ/picmi_hall_rz_core_smoke.py
Test/HallRZ/picmi_hall_rz_pic_benchmark.py
  EB array factory 增加 nr/nz 参数。
  AMR benchmark 非 scalar boundary 时逐 level 上传 shape 为：
    lev0 = 70 x 80
    lev1 = 140 x 160
    lev2 = 280 x 320
  AMR benchmark 当前只上传 Python EB arrays，不上传 single-level physical Python arrays。
```

当前 Stage 3.3 支持范围：

```text
单层:
  仍兼容 hallrz.set_eb_neumann(g) / hallrz.set_eb_robin(a,b,f) 默认 lev=0。

三层 AMR:
  EB Neumann:
    可使用 scalar/default EB Neumann；
    也可使用 hallrz.set_eb_neumann(g, lev=0/1/2) 逐 level 上传。
  EB Robin:
    必须使用 hallrz.set_eb_robin(a,b,f, lev=0/1/2) 逐 level 上传。
  physical domain Python arrays:
    z-hi Robin、r-hi Robin、r-lo Robin、r-lo Dirichlet、inlet Dirichlet 仍是单层 shape，
    AMR path 暂不支持。
```

Build：

```text
python -m py_compile Python/pywarpx/hallrz.py \
  Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
  Test/HallRZ/picmi_hall_rz_core_smoke.py
result: pass

cmake --build build -j 8 --target pip_install_nodeps
result: pass
```

接口探针：

```text
pywarpx.geometry.dims = "RZ"
hallrz.clear_eb_neumann()
hallrz.clear_eb_robin()
hallrz.set_eb_neumann(np.ones((2,3)), lev=1)
hallrz.has_eb_neumann() / has_eb_neumann(0/1/2)
hallrz.clear_eb_neumann(1)
hallrz.set_eb_robin(..., lev=2)
hallrz.has_eb_robin() / has_eb_robin(2)
hallrz.clear_eb_robin()

result:
  initial False False
  neumann any/0/1/2 True False True False
  after clear lev1 False
  robin any/2 True True
  final False
```

测试：

```text
stage3_3_single_level_python_eb_neumann_smoke
  command summary = Test/HallRZ/hallrz_stage3_3_single_level_python_eb_neumann_smoke.json
  log             = Test/HallRZ/hallrz_stage3_3_single_level_python_eb_neumann_smoke.log
  result: pass
  nprocs=1
  avg_step_wall_time_s = 0.0035523829865269363
  requested_mcl=0, actual_nmg_levels=1
  Final Iter=7
  resid/resid0=1.247188652e-11
  q_EB_diff_RZ=0, epsilon_q=0
  N_NaN=0, N_Inf=0, particle NaN/Inf=0

stage3_3_amr_python_eb_neumann_smoke
  command summary = Test/HallRZ/hallrz_stage3_3_amr_python_eb_neumann_smoke.json
  log             = Test/HallRZ/hallrz_stage3_3_amr_python_eb_neumann_smoke.log
  result: pass
  nprocs=1
  python_eb_levels=[0,1,2], python_physical_boundary_arrays=false
  boxes = 1 / 1 / 1
  avg_step_wall_time_s = 0.2225560850056354
  level 0: requested_mcl=0, actual_nmg_levels=1, Final Iter=6, resid/resid0=4.23092351e-11
  level 1: requested_mcl=1, actual_nmg_levels=2, Final Iter=4, resid/resid0=3.113072487e-12
  level 2: requested_mcl=2, actual_nmg_levels=2, Final Iter=4, resid/resid0=1.773569538e-12
  q_EB_diff_RZ=0 on all levels
  N_NaN=0, N_Inf=0, particle NaN/Inf=0

stage3_3_amr_python_eb_robin_smoke
  command summary = Test/HallRZ/hallrz_stage3_3_amr_python_eb_robin_smoke.json
  log             = Test/HallRZ/hallrz_stage3_3_amr_python_eb_robin_smoke.log
  result: pass
  nprocs=1
  python_eb_levels=[0,1,2], python_physical_boundary_arrays=false
  boxes = 1 / 1 / 1
  avg_step_wall_time_s = 0.22768238600110635
  level 0: requested_mcl=0, actual_nmg_levels=1, Final Iter=6, resid/resid0=3.902408319e-11
  level 1: requested_mcl=1, actual_nmg_levels=2, Final Iter=4, resid/resid0=3.108358137e-12
  level 2: requested_mcl=2, actual_nmg_levels=2, Final Iter=4, resid/resid0=1.761449455e-12
  q_EB_diff_RZ=0 on all levels
  N_NaN=0, N_Inf=0, particle NaN/Inf=0

stage3_3_amr_python_eb_robin_mpi_np4_smoke
  command summary = Test/HallRZ/hallrz_stage3_3_amr_python_eb_robin_mpi_np4_smoke.json
  log             = Test/HallRZ/hallrz_stage3_3_amr_python_eb_robin_mpi_np4_smoke.log
  result: pass
  nprocs=4 sharing one GPU; correctness-only, not speed baseline
  python_eb_levels=[0,1,2], python_physical_boundary_arrays=false
  boxes = 4 / 4 / 4
  level 0: requested_mcl=0, actual_nmg_levels=1, Final Iter=6, resid/resid0=3.741312365e-11
  level 1: requested_mcl=1, actual_nmg_levels=2, Final Iter=4, resid/resid0=3.145270783e-12
  level 2: requested_mcl=2, actual_nmg_levels=2, Final Iter=4, resid/resid0=1.634930208e-12
  q_EB_diff_RZ=0 on all levels
  N_NaN=0, N_Inf=0, particle NaN/Inf=0

stage3_3_amr_incomplete_python_eb_guard_expected_abort
  log = Test/HallRZ/hallrz_stage3_3_amr_incomplete_python_eb_guard_expected_abort.log
  input: AMR Neumann mode, only hallrz.set_eb_neumann(..., lev=0)
  result: pass expected-abort
  message:
    HallRZ AMR Python EB Neumann data must be provided for every AMR level,
    or cleared to use scalar/default EB Neumann data.
```

当前 Stage 3.3 判断：

```text
Phase 3.3 per-level EB geometry and EB boundary data: pass for current GPU-tree smoke scope.

已满足：
  - 单层 lev=0 API 保持兼容；
  - 三层 AMR EB Neumann per-level arrays 可用；
  - 三层 AMR EB Robin per-level arrays 可用；
  - Neumann-equivalent Robin 与 Neumann 的迭代、残差和场量同量级一致；
  - np=4 shared-GPU MPI correctness smoke pass；
  - incomplete per-level EB data 会明确 abort，不会静默复用 lev0 数据；
  - physical domain Python arrays 仍被 AMR guard 禁止。

未完成，不能视为 Stage 3 完成：
  - per-level physical domain Python arrays 尚未实现；
  - coarse/fine continuity quantitative diagnostic 尚未实现；
  - Stage 3.4 E-field writeback 专项验收尚未开始；
  - 1000 步完整 PIC gate 尚未跑。
```

### 2026-06-07 Stage 3.4 AMR HallRZ E-field writeback 执行记录

本轮继续 Stage 3.4，目标是确认每个 AMR level 的 `phi` 都写回该 level 的 staggered/Yee
`Er/Ez`，并把 benchmark 验收从 level 0 扩展到所有 active AMR levels。

已完成的 Stage 3.4 修改：

```text
Test/HallRZ/picmi_hall_rz_pic_benchmark.py
  新增 per-level field summary:
    field_levels[lev].fields.rho_fp
    field_levels[lev].fields.phi_fp
    field_levels[lev].fields.Er_fp
    field_levels[lev].fields.Ez_fp
  旧 fields 字段仍保留为 level 0，保持既有单层报告兼容。
  checks.fields_finite 现在覆盖所有 field_levels。
  checks.field_levels_finite 显式记录所有 level 字段 finite 状态。
```

代码审查确认：

```text
Source/FieldSolver/ElectrostaticSolvers/LabFrameExplicitES.cpp
  AMR path 对 lev=0,1,2 逐层调用：
    HallRZPoissonSolver::ComputePhiAndE(..., Efield_fp[lev], ..., lev)

Source/FieldSolver/ElectrostaticSolvers/HallRZPoissonSolver.cpp
  ComputeStaggeredE() 保持 Stage 1 的 Yee/staggered 布局：
    Efield_fp[0] = Er, ixType IntVect(0,1)
    Efield_fp[2] = Ez, ixType IntVect(1,0)
  covered/non-computational edge 保护仍存在：
    Er 只有 ls(i,j)<=0 && ls(i+1,j)<=0 时写有限差分，否则写 0.0
    Ez 只有 ls(i,j)<=0 && ls(i,j+1)<=0 时写有限差分，否则写 0.0
```

测试：

```text
python -m py_compile Test/HallRZ/picmi_hall_rz_pic_benchmark.py
result: pass

stage3_4_single_level_efield_stats_smoke
  summary = Test/HallRZ/hallrz_stage3_4_single_level_efield_stats_smoke.json
  log     = Test/HallRZ/hallrz_stage3_4_single_level_efield_stats_smoke.log
  result: pass
  nprocs=1
  avg_step_wall_time_s = 0.002741495001828298
  checks.fields_finite = true
  checks.field_levels_finite = true
  field_levels = 1
  lev0:
    phi_finite=true, Er_finite=true, Ez_finite=true
    Er_max_abs=5.15309520328, Ez_max_abs=9.7880329308
  HallRZ particle summary:
    N_particle_NaN=0, N_particle_Inf=0, N_inside_covered_EB=0
  warnings:
    No recorded warnings.

stage3_4_amr_robin_efield_stats_smoke
  summary = Test/HallRZ/hallrz_stage3_4_amr_robin_efield_stats_smoke.json
  log     = Test/HallRZ/hallrz_stage3_4_amr_robin_efield_stats_smoke.log
  result: pass
  nprocs=1
  avg_step_wall_time_s = 0.2150586069910787
  checks.fields_finite = true
  checks.field_levels_finite = true
  field_levels = 3
  level 0:
    actual_nmg_levels=1, Final Iter=6, N_NaN=0, N_Inf=0
    phi_finite=true, Er_finite=true, Ez_finite=true
    Er_max_abs=5.15296369979, Ez_max_abs=9.7880245471
  level 1:
    actual_nmg_levels=2, Final Iter=4, N_NaN=0, N_Inf=0
    phi_finite=true, Er_finite=true, Ez_finite=true
    Er_max_abs=6.61668687188, Ez_max_abs=10.2255431312
  level 2:
    actual_nmg_levels=2, Final Iter=4, N_NaN=0, N_Inf=0
    phi_finite=true, Er_finite=true, Ez_finite=true
    Er_max_abs=8.39049089271, Ez_max_abs=10.6656084999
  HallRZ particle summary:
    N_particle_NaN=0, N_particle_Inf=0, N_inside_covered_EB=0
  warnings:
    No recorded warnings.

stage3_4_amr_robin_efield_stats_mpi_np4_smoke
  summary = Test/HallRZ/hallrz_stage3_4_amr_robin_efield_stats_mpi_np4_smoke.json
  log     = Test/HallRZ/hallrz_stage3_4_amr_robin_efield_stats_mpi_np4_smoke.log
  result: pass
  nprocs=4 sharing one GPU; correctness-only, not speed baseline
  checks.fields_finite = true
  checks.field_levels_finite = true
  field_levels = 3
  boxes = 4 / 4 / 4
  level 0:
    actual_nmg_levels=1, Final Iter=6, N_NaN=0, N_Inf=0
    phi_finite=true, Er_finite=true, Ez_finite=true
    Er_max_abs=5.15296372008, Ez_max_abs=9.78802492225
  level 1:
    actual_nmg_levels=2, Final Iter=4, N_NaN=0, N_Inf=0
    phi_finite=true, Er_finite=true, Ez_finite=true
    Er_max_abs=6.61668687705, Ez_max_abs=10.2255433509
  level 2:
    actual_nmg_levels=2, Final Iter=4, N_NaN=0, N_Inf=0
    phi_finite=true, Er_finite=true, Ez_finite=true
    Er_max_abs=8.39049091908, Ez_max_abs=10.6656087203
  HallRZ particle summary:
    N_particle_NaN=0, N_particle_Inf=0, N_inside_covered_EB=0
  warnings:
    No recorded warnings.
```

当前 Stage 3.4 判断：

```text
Phase 3.4 AMR HallRZ E-field writeback: pass for current GPU-tree smoke scope.

已满足：
  - level 0/1/2 均执行 phi -> staggered Er/Ez writeback；
  - JSON summary 对每个 level 的 rho/phi/Er/Ez 做 finite 统计；
  - AMR EB Robin np=1 与 np=4 shared-GPU smoke 均显示每层 Er/Ez finite；
  - C++ covered/non-computational edge 保护代码未被修改，仍在 protected edge 写 0.0；
  - 粒子 summary 中 N_inside_covered_EB=0，未观察到 covered 区域 gather 异常。

未完成，不能视为 Stage 3 完成：
  - coarse/fine continuity quantitative diagnostic 尚未实现；
  - 1000 步完整 PIC gate 尚未跑；
  - per-level physical domain Python arrays 仍未实现。
```

### 2026-06-07 Stage 3.5 完整 PIC 验收执行记录

本轮继续 Stage 3.5，只做当前 GPU tree 的完整 PIC gate、shared-GPU MPI 正确性 gate 和计划记录。
不修改 AMReX。

已完成的 Stage 3.5 benchmark 基础设施补充：

```text
Test/HallRZ/picmi_hall_rz_pic_benchmark.py
  新增 coarse/fine artificial Dirichlet continuity diagnostic:
    collect_coarse_fine_phi_continuity(args, finest_level)
    _node_bilinear_interp(coarse_phi, fine_i, fine_j)
    _error_stats(error)

  检查对象：
    fine level 1 的 r-hi / z-hi artificial coarse Dirichlet patch face
      对 coarse level 0 的 node-bilinear interpolation；
    fine level 2 的 r-hi / z-hi artificial coarse Dirichlet patch face
      对 coarse level 1 的 node-bilinear interpolation。

  容差：
    COARSE_FINE_PHI_TOL = 1.0e-10

  JSON summary 新增：
    coarse_fine_phi_continuity
    checks.coarse_fine_phi_continuity
```

说明：

- 该 diagnostic 验证 Stage 3.2/3.3 中 fine patch 人工高边界使用 coarse `phi` 插值作为
  Dirichlet 数据的实际连续性。
- 它不验证完整全局 AMR composite Poisson operator；当前 HallRZ AMR 仍是 level-by-level solve。
- 当前 benchmark 的 `field_levels` 已覆盖每个 active AMR level 的 `rho/phi/Er/Ez` finite 统计。
- C++ `ComputePhiAndE()` 仍保留 `rho_before` / `rho_after_diff` 计算逻辑，并在 verbose>=2 时打印
  `rho_mutation_inf`；本轮默认 gate 日志没有把该值作为 JSON 断言输出，因此不能把“独立
  solve 前后 rho mutation JSON gate”写成已完成。

测试与结果：

```text
python -m py_compile Test/HallRZ/picmi_hall_rz_pic_benchmark.py
result: pass

stage3_5_cf_continuity_probe
  command:
    python Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
      --case stage3_5_cf_continuity_probe \
      --max-steps 1 --chunk-steps 1 \
      --summary-json Test/HallRZ/hallrz_stage3_5_cf_continuity_probe.json \
      --hall-rz-amr-enable \
      --hall-rz-amr-r1 0.060 --hall-rz-amr-z1 0.060 \
      --hall-rz-amr-r2 0.055 --hall-rz-amr-z2 0.050 \
      --hall-rz-max-coarsening-levels 0 1 2 \
      --warpx-max-grid-size 280 \
      --eb-bc-mode robin \
      --amrex-the-arena-init-size 0
  result: pass
  checks.pass = true
  coarse_fine_phi_continuity.enabled = true
  coarse_fine_phi_continuity.max_abs = 0.0
  tolerance = 1.0e-10

stage3_5_single_level_neumann_mcl0_full
  command:
    python Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
      --case stage3_5_single_level_neumann_mcl0_full \
      --max-steps 1000 --chunk-steps 50 \
      --summary-json Test/HallRZ/hallrz_stage3_5_single_level_neumann_mcl0_full.json \
      --hall-rz-max-coarsening-level 0
  result: pass
  nprocs = 1
  checks.pass = true
  avg_step_wall_time_s = 0.03060207515998627
  total_step_wall_time_s = 30.60207515998627
  particles = electrons 1800, protons 1800
  field_levels = 1
  final log:
    amr_level=0, requested_mcl=0, actual_nmg_levels=1
    Final Iter=1, resid/resid0=2.476157648e-12
    q_EB_diff_RZ=0, N_NaN=0, N_Inf=0
    N_particle_NaN=0, N_particle_Inf=0
    N_out_of_domain=0, N_inside_covered_EB=0
    No recorded warnings.

stage3_5_amr_neumann_full
  command:
    python Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
      --case stage3_5_amr_neumann_full \
      --max-steps 1000 --chunk-steps 50 \
      --summary-json Test/HallRZ/hallrz_stage3_5_amr_neumann_full.json \
      --hall-rz-amr-enable \
      --hall-rz-amr-r1 0.060 --hall-rz-amr-z1 0.060 \
      --hall-rz-amr-r2 0.055 --hall-rz-amr-z2 0.050 \
      --hall-rz-max-coarsening-levels 0 1 2 \
      --warpx-max-grid-size 280
  result: pass
  nprocs = 1
  checks.pass = true
  avg_step_wall_time_s = 0.23148196476095473
  total_step_wall_time_s = 231.48196476095472
  coarse_fine_phi_continuity.max_abs = 0.0
  tolerance = 1.0e-10
  final log:
    level 0: requested_mcl=0, actual_nmg_levels=1, Final Iter=1,
             resid/resid0=3.254382235e-12, N_NaN=0, N_Inf=0
    level 1: requested_mcl=1, actual_nmg_levels=2, Final Iter=4,
             resid/resid0=3.118586409e-12, N_NaN=0, N_Inf=0
    level 2: requested_mcl=2, actual_nmg_levels=2, Final Iter=4,
             resid/resid0=1.757717483e-12, N_NaN=0, N_Inf=0
    N_particle_NaN=0, N_particle_Inf=0
    N_out_of_domain=0, N_inside_covered_EB=0
    No recorded warnings.

stage3_5_amr_robin_full
  command:
    python Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
      --case stage3_5_amr_robin_full \
      --max-steps 1000 --chunk-steps 50 \
      --summary-json Test/HallRZ/hallrz_stage3_5_amr_robin_full.json \
      --hall-rz-amr-enable \
      --hall-rz-amr-r1 0.060 --hall-rz-amr-z1 0.060 \
      --hall-rz-amr-r2 0.055 --hall-rz-amr-z2 0.050 \
      --hall-rz-max-coarsening-levels 0 1 2 \
      --warpx-max-grid-size 280 \
      --eb-bc-mode robin
  result: pass
  nprocs = 1
  checks.pass = true
  avg_step_wall_time_s = 0.23129732540107215
  total_step_wall_time_s = 231.29732540107216
  coarse_fine_phi_continuity.max_abs = 0.0
  tolerance = 1.0e-10
  final log:
    level 0: requested_mcl=0, actual_nmg_levels=1, Final Iter=1,
             resid/resid0=3.418355099e-12, N_NaN=0, N_Inf=0
    level 1: requested_mcl=1, actual_nmg_levels=2, Final Iter=4,
             resid/resid0=3.120570717e-12, N_NaN=0, N_Inf=0
    level 2: requested_mcl=2, actual_nmg_levels=2, Final Iter=4,
             resid/resid0=1.751361645e-12, N_NaN=0, N_Inf=0
    N_particle_NaN=0, N_particle_Inf=0
    N_out_of_domain=0, N_inside_covered_EB=0
    No recorded warnings.

stage3_5_amr_robin_mpi_np4_50step
  command:
    OMP_NUM_THREADS=1 mpiexec -np 4 python Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
      --case stage3_5_amr_robin_mpi_np4_50step \
      --max-steps 50 --chunk-steps 10 \
      --summary-json Test/HallRZ/hallrz_stage3_5_amr_robin_mpi_np4_50step.json \
      --hall-rz-amr-enable \
      --hall-rz-amr-r1 0.060 --hall-rz-amr-z1 0.060 \
      --hall-rz-amr-r2 0.055 --hall-rz-amr-z2 0.050 \
      --hall-rz-max-coarsening-levels 0 1 2 \
      --warpx-max-grid-size 140 \
      --eb-bc-mode robin
  result: pass
  nprocs = 4, one shared GPU, correctness-only and not speed baseline
  boxes = 4 / 4 / 4
  checks.pass = true
  avg_step_wall_time_s = 8.155031573919695
  total_step_wall_time_s = 407.7515786959848
  coarse_fine_phi_continuity.max_abs = 0.0
  tolerance = 1.0e-10
  final log:
    level 0: requested_mcl=0, actual_nmg_levels=1, Final Iter=0,
             resid/resid0=5.537568026e-11, N_NaN=0, N_Inf=0
    level 1: requested_mcl=1, actual_nmg_levels=2, Final Iter=4,
             resid/resid0=3.112983853e-12, N_NaN=0, N_Inf=0
    level 2: requested_mcl=2, actual_nmg_levels=2, Final Iter=4,
             resid/resid0=1.770985524e-12, N_NaN=0, N_Inf=0
    N_particle_NaN=0, N_particle_Inf=0
    N_out_of_domain=0, N_inside_covered_EB=0
    No recorded warnings.
```

速度对照：

```text
已知旧记录：
  Stage 0 old baseline                 = 0.193826259517 s/step
  Stage 1 recommended MCL=0 baseline   = 0.029709197016 s/step
  Stage 2 single-level 1000-step gate  = 0.029417214385 s/step

本轮：
  Stage 3.5 single-level MCL=0         = 0.03060207515998627 s/step
    vs Stage 1 recommended baseline: +3.01%
    vs Stage 2 single-level gate:     +4.03%
    vs Stage 0 old baseline:         -84.21%

  Stage 3.5 AMR Neumann 1000-step      = 0.23148196476095473 s/step
  Stage 3.5 AMR Robin 1000-step        = 0.23129732540107215 s/step
    Robin vs Neumann: -0.08%

解释：
  - 单层路径没有超过 Stage 1 +10% 退化门槛。
  - Neumann 和 Robin 三层 AMR 速度基本相同，说明 EB Robin API/BC 本身不是当前性能退化源。
  - 三层 level-by-level AMR 当前比单层 MCL=0 benchmark 慢约 7.56 倍；
    这不是等效全 finest 单层的加速对照。当前 benchmark 固定 NR/NZ，尚未设计 full-finest
    single-level comparator，因此不能把“三层 AMR 相比等效全 finest 单层有加速”写成已验证。
  - np=4 shared-GPU 结果只用于 MPI correctness，不用于速度评估。
```

当前 Stage 3.5 判断：

```text
Phase 3.5: pass for current GPU-tree Stage 3 gate scope.

已满足：
  - 单层 Stage 1 regression 1000 步通过，速度相对 Stage 1 baseline +3.01%，未超过 +10% 门槛；
  - 三层 AMR Neumann 1000 步通过；
  - 三层 AMR Robin 1000 步通过；
  - 三层 AMR Robin np=4 shared-GPU MPI 正确性 gate 通过；
  - 每个 active level 的 rho/phi/Er/Ez finite；
  - coarse/fine artificial Dirichlet continuity max_abs=0.0 <= 1.0e-10；
  - step-end Poisson log 每层 residual 达到约 1e-11 到 1e-12，N_NaN=0，N_Inf=0；
  - 粒子统计 N_particle_NaN=0，N_particle_Inf=0，N_out_of_domain=0，
    N_inside_covered_EB=0；
  - EB Neumann 和 EB Robin 三层 AMR 速度等价，未观察到 Robin-specific 性能退化；
  - ComputeStaggeredE() 的 covered/non-computational edge 保护未被改动。

未覆盖或仍需用户审查：
  - 本轮没有运行 CPU backend PIC gate。此前用户允许当前 GPU tree 阶段用 np=4 shared-GPU
    作为多 MPI correctness gate，不代表 CPU backend 已完成。
  - 未实现 full-finest single-level comparator；因此三层 AMR 的“相对等效全 finest 加速”尚未
    验证。
  - per-level physical domain Python arrays 仍未实现；当前人工 AMR high faces 使用 coarse
    Dirichlet，其他 faces 仍按物理边界处理。
  - JSON summary 尚未包含独立 `rho_mutation_inf` 断言；若后续要把 rho 不被 solve 修改作为
    自动 gate，需要把 C++ verbose 诊断或 Python-side before/after check 纳入 benchmark。
  - Stage 3 是否作为整体完成，仍需要用户审查本记录后确认。
```

### Phase 3.6：三层 AMR vs 等效全 finest 单层 200-step 扫描

目标：

- 在同一物理区域内，对比三层 static rectangular AMR 与等效全 finest 单层的速度和场误差。
- 该 phase 只补充 benchmark infrastructure 和性能/误差证据，不改变 C++ HallRZ solver 逻辑。
- 扫描从较小网格到较大网格，观察速度、误差和收敛趋势；不能只跑单个网格点。

网格族采用用户确认的 option 2：

```text
s = 5, 6, 7, 8, 9

AMR level 0: nr = 14*s, nz = 16*s
AMR level 1: refinement ratio 2
AMR level 2: refinement ratio 2

等效全 finest 单层: nr = 56*s, nz = 64*s
```

采用该网格族的原因：

- Stage 3.6 原先候选 `k=5,10,15,20,25` 中，`k=5,15,25` 会导致
  `lob=0.035` 不落在 AMR level-0 cell face 上，不满足当前 HallRZ static AMR 几何对齐输入
  contract。
- `s` 网格族的 level-0 cell size 为 `dr=dz=0.005/s`，因此
  `lob=7*s`、`hib=10*s`、`out=8*s`、`r1=z1=12*s`、`r2=11*s`、`z2=10*s`
  都落在 level-0 cell face 上；level 1/2 由于 refinement ratio 为 2 自动保持对齐。
- 等效全 finest 单层 cell size 与 AMR level 2 cell size 相同，用于速度和场误差对照。

固定几何：

```text
rmin = 0.0
rmax = 0.070
zmin = 0.0
zmax = 0.080
lob  = 0.035
hib  = 0.050
out  = 0.040
r1,z1 = 0.060,0.060
r2,z2 = 0.055,0.050
```

运行设置：

```text
steps = 200
chunk_steps = 20
full-finest: hall_rz_max_coarsening_level = 3
AMR: hall_rz_max_coarsening_levels = 0 1 2
EB BC: robin, using Neumann-equivalent raw EB-FVM Robin a=0,b=1,f=gn
particle mode: fixed deterministic particles
```

Full-finest MCL 选择说明：

- Stage 3.6 smoke 已确认 `s=5` full-finest 单层在 `MCL=0` 和 `MCL=1` 下会因为
  `MLMG failed` 退出；`MCL=30` 会在当前 1-GPU 环境触发深 coarsening 的 GPU arena OOM。
- `MCL=2` 的 `s=5` full-finest 1-step smoke 可收敛，但正式扫描推进到 `s=8` 时，
  full-finest 初始 solve 在 `MCL=2` 下再次 `MLMG failed`。
- `MCL=3` 的 `s=8` 和 `s=9` full-finest 1-step smoke 均可收敛并通过 field/particle checks。
- 因此 full-finest comparator 使用显式 `MCL=3`。这不是 AMR level 数，而是单层 full-finest
  solve 的 AMReX MG upper bound；JSON summary 必须记录该值。不得回退到默认深 coarsening。

fixed deterministic particles 规则：

- 每个 species 使用同一套固定 `(r,z)` 粒子坐标和权重，默认 `60 x 30` 个粒子。
- 粒子数不随网格大小变化，避免 full-finest 与 AMR 之间因为 PICMI 随网格布点而改变粒子总数。
- fixed particles 必须在初始 HallRZ E solve 前进入 particle container。实现上使用
  `pywarpx.callbacks.beforeInitEsolve` 注入，并保留 init-only 路径的 post-init fallback。
- 不允许用 `ParticleListDistribution` 把长数组塞进 ParmParse/argv；这会触发 OpenMPI singleton
  `Argument list too long`，不能作为正式 benchmark 路径。

新增/使用文件：

```text
Test/HallRZ/hallrz_stage3_6_amr_fullfinest_scan.py
Test/HallRZ/picmi_hall_rz_pic_benchmark.py
Test/HallRZ/picmi_hall_rz_core_smoke.py
```

输出文件：

```text
Test/HallRZ/hallrz_stage3_6_s<s>_full_finest.json
Test/HallRZ/hallrz_stage3_6_s<s>_full_finest.npz
Test/HallRZ/hallrz_stage3_6_s<s>_full_finest.log
Test/HallRZ/hallrz_stage3_6_s<s>_amr.json
Test/HallRZ/hallrz_stage3_6_s<s>_amr.npz
Test/HallRZ/hallrz_stage3_6_s<s>_amr.log
Test/HallRZ/hallrz_stage3_6_amr_vs_fullfinest.csv
Test/HallRZ/hallrz_stage3_6_amr_vs_fullfinest.md
```

速度指标：

- full-finest 与 AMR 的 `avg_step_wall_time_s`。
- 去掉首个 chunk 后的 steady `s/step`，用于减少初始化后首段热身噪声。
- `speedup_steady = full_finest_steady_s_per_step / amr_steady_s_per_step`。
- 若 `speedup_steady <= 1`，不能声称 AMR 有速度优势；必须报告具体数据和可能开销来源。

误差指标：

- 每个 AMR level 的 `phi_fp`、`Er_fp`、`Ez_fp` 与 full-finest 单层参考场比较。
- 比较时把 full-finest 参考场按物理坐标插值到 AMR level 的节点/交错位置。
- AMR coarse level 中被 finer level 覆盖的 lower-left 区域不计入该 coarse level 的 owned-region
  误差，避免重复统计。
- 报告每个 field 的 `linf_abs`、`l2_abs`、`linf_rel`、`l2_rel`，并给出 global owned-region 误差。
- Phase 3.6 首轮不预设绝对误差 pass/fail 阈值；先要求 finite、趋势可解释，并由用户根据扫描表
  决定后续 AMR 精度门槛。

验收标准：

- `python -m py_compile` 覆盖三个 benchmark/scan 脚本并通过。
- 至少一个 `s=5` full-finest 和 AMR 短 smoke 能完成，`checks.pass=true`。
- 完整 `s=5..9`、每点 200 步扫描完成；若某个大网格因资源限制失败，必须保留已完成点并报告
  失败命令、错误摘要和最大已完成网格，不得把不完整扫描写成通过。
- 每个完成点：
  - `checks.pass=true`；
  - field finite，particle count positive；
  - AMR coarse/fine artificial Dirichlet continuity 达到已有 `1e-10` gate；
  - log 中 HallRZ 每层 residual、requested MCL、actual NMG levels 可复查；
  - 无 NaN/Inf；
  - fixed particles 在初始 E solve 前加载，不能出现 empty-particle 初始 solve 失败。
- 生成 CSV 和 Markdown 汇总表，包含每个 `s` 的速度、speedup 和 global field errors。
- Phase 3.6 完成报告必须明确：
  - 与 Stage 3.5 三层 AMR 速度的关系；
  - 与等效 full-finest 单层速度的 speedup；
  - `phi/Er/Ez` 误差；
  - 是否观察到网格加密后的误差趋势；
  - residual risks 和下一步建议。

Stage 3.6 执行记录（2026-06-07）：

```text
py_compile:
  python -m py_compile \
    Test/HallRZ/picmi_hall_rz_core_smoke.py \
    Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
    Test/HallRZ/hallrz_stage3_6_amr_fullfinest_scan.py
  result: pass

smoke:
  - full-finest s=5, fixed particles, EB Robin, MCL=2: pass
  - AMR s=5, fixed particles, EB Robin, MCL=0 1 2: pass
  - scan driver s=5, steps=1: pass and generated CSV/Markdown

MCL correction:
  - full-finest s=5 MCL=0/1 failed with MLMG failed.
  - full-finest s=5 MCL=30 failed with GPU arena OOM.
  - full-finest s=8 MCL=2 failed with MLMG failed.
  - full-finest s=8/s=9 MCL=3 1-step smoke passed.
  - final full-finest comparator therefore uses explicit MCL=3.

final scan command:
  python Test/HallRZ/hallrz_stage3_6_amr_fullfinest_scan.py \
    --s-values 5 6 7 8 9 \
    --steps 200 --chunk-steps 20 \
    --max-grid-size 280 \
    --prefix hallrz_stage3_6_mcl3

outputs:
  Test/HallRZ/hallrz_stage3_6_mcl3_amr_vs_fullfinest.csv
  Test/HallRZ/hallrz_stage3_6_mcl3_amr_vs_fullfinest.md
```

最终扫描表：

```text
s  full grid  AMR L0 grid  full steady s/step  AMR steady s/step  speedup  phi L2 rel  Er L2 rel   Ez L2 rel
5  280x320    70x80        0.040318            0.224196           0.179834 0.120925    0.00671426  0.00108244
6  336x384    84x96        0.0415205           0.234780           0.176848 0.110881    0.00543335  0.000871099
7  392x448    98x112       0.0425475           0.225439           0.188732 0.0949384   0.0167121   0.0136854
8  448x512    112x128      0.045407            0.267249           0.169905 0.0966322   0.00430371  0.000703157
9  504x576    126x144      0.0530612           0.274976           0.192967 0.0912700   0.00392066  0.000656792
```

Stage 3.6 判断：

```text
Correctness/infrastructure: pass for the 200-step scan scope.

已满足：
  - s=5..9 full-finest 与 AMR 均完成 200 steps；
  - 每个 JSON summary 的 checks.pass=true；
  - fixed particles 每个 species 为 1800，full-finest 与 AMR 一致；
  - AMR coarse/fine artificial Dirichlet continuity max_abs=0.0；
  - field finite、particle count positive、无 NaN/Inf；
  - log 中可复查 requested_mcl、actual_nmg_levels、Final Iter、resid/resid0。

速度结论：
  - 当前三层 level-by-level AMR 没有比等效 full-finest 单层快；
  - speedup_steady 约 0.17 到 0.19，即 AMR 比 full-finest 慢约 5.2 到 5.9 倍；
  - 因此不能把 Stage 3.6 写成 AMR speedup 成功。后续若要速度收益，必须先分析
    level-by-level 三次 solve、level-0 MCL=0、coarse/fine boundary fill、重复 setup/cache 等开销。

误差结论：
  - global phi L2 rel 从 0.1209 下降到 0.0913，整体随加密下降但不单调严格；
  - Er/Ez global L2 rel 大多数点为 1e-3 到 1e-2，s=7 有局部异常偏高；
  - phi Linf rel 约 0.8，说明存在局部最大误差或参考值尺度问题，不能只看 L2 判断精度；
  - Phase 3.6 只记录误差，不把该误差表作为最终 AMR 精度验收阈值。

未解决：
  - full-finest 需要显式 MCL=3 才覆盖 s=8/s=9；MCL policy 后续仍需独立设计。
  - AMR 当前采用 level-by-level independent solve + coarse Dirichlet artificial faces，
    不是真正 coupled composite AMR solve；速度和误差都反映这一首版算法限制。
```

## 7. Stage 4：单层收敛与 AMR 成本/准确性 Debug

Stage 4 目标：

- 先证明单层 full-finest HallRZ 的收敛行为和准确行为可解释，再讨论三层 AMR。
- 三层 AMR 必须先算准，再在大网格 `s=8/9` 相比 full-finest 达到 `speedup_steady >= 3`。
- Stage 4 分两阶段推进：先诊断当前 level-by-level independent solve 框架；若证据显示该框架
  天然达不到速度/精度目标，再单独设计 composite/coupled AMR solve。

Stage 4 新增 debug infrastructure（2026-06-07）：

```text
Test/HallRZ/picmi_hall_rz_core_smoke.py
  - build_sim() 新增 static-rho 参数：
    static_rho_test, static_rho_mode, static_rho_amp,
    static_rho_lambda_r, static_rho_lambda_z

Test/HallRZ/picmi_hall_rz_pic_benchmark.py
  - CLI 新增 --static-rho-test / --static-rho-mode / --static-rho-amp /
    --static-rho-lambda-r / --static-rho-lambda-z
  - JSON summary.solver 记录 static-rho 设置

Test/HallRZ/hallrz_stage4_single_level_mcl_scan.py
  - 扫描单层 full-finest 的 s/MCL 收敛矩阵
  - 非零退出不终止整个矩阵；记录 mlmg_failed / gpu_oom / sigabrt 等分类
  - 从 log 提取 actual_nmg_levels、Final Iter、resid/resid0、rho_mutation_inf

Test/HallRZ/hallrz_stage4_amr_cost_model.py
  - 读取 Stage 3.6 mcl3 结果
  - 计算 full-finest 与 AMR level-by-level solve 的 rho-cell count ratio
  - 与实测 speedup 和目标 3x 对比
```

Stage 4 第一批命令：

```text
python -m py_compile \
  Test/HallRZ/picmi_hall_rz_core_smoke.py \
  Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
  Test/HallRZ/hallrz_stage4_single_level_mcl_scan.py \
  Test/HallRZ/hallrz_stage4_amr_cost_model.py \
  Test/HallRZ/hallrz_stage3_6_amr_fullfinest_scan.py

python Test/HallRZ/hallrz_stage4_single_level_mcl_scan.py \
  --s-values 5 6 7 8 9 \
  --mcls 0 1 2 3 4 \
  --steps 1 --chunk-steps 1 \
  --prefix hallrz_stage4_single_mcl_static

python Test/HallRZ/hallrz_stage4_amr_cost_model.py \
  --input-prefix hallrz_stage3_6_mcl3 \
  --output-prefix hallrz_stage4_amr_cost_model
```

Stage 4 第一批输出：

```text
Test/HallRZ/hallrz_stage4_single_mcl_static.csv
Test/HallRZ/hallrz_stage4_single_mcl_static.md
Test/HallRZ/hallrz_stage4_amr_cost_model.csv
Test/HallRZ/hallrz_stage4_amr_cost_model.md
```

单层 static-rho MCL 收敛矩阵：

```text
rho source: static_rho_test, mode=1, amp=1e-10, lambda_r=0.004, lambda_z=0.006
particle mode: fixed
EB BC: Robin Neumann-equivalent
steps: 1

s=5, full=280x320:
  MCL 0/1: MLMG failed
  MCL 2: pass, actual_nmg=3, iter=7,  resid/resid0=1.10e-11
  MCL 3: pass, actual_nmg=4, iter=6,  resid/resid0=5.12e-12
  MCL 4: pass, actual_nmg=4, iter=6,  resid/resid0=1.82e-12

s=6, full=336x384:
  MCL 0/1: MLMG failed
  MCL 2: pass, actual_nmg=3, iter=12, resid/resid0=8.72e-12
  MCL 3: pass, actual_nmg=4, iter=10, resid/resid0=8.81e-12
  MCL 4: pass, actual_nmg=5, iter=11, resid/resid0=1.10e-11

s=7, full=392x448:
  MCL 0/1/2: MLMG failed
  MCL 3: pass, actual_nmg=4, iter=9, resid/resid0=7.85e-12
  MCL 4: pass, actual_nmg=4, iter=9, resid/resid0=7.24e-12

s=8, full=448x512:
  MCL 0/1: MLMG failed
  MCL 2: pass, actual_nmg=3, iter=32, resid/resid0=6.56e-12
  MCL 3: pass, actual_nmg=4, iter=11, resid/resid0=5.17e-12
  MCL 4: pass, actual_nmg=5, iter=10, resid/resid0=6.88e-12

s=9, full=504x576:
  MCL 0/1/2: MLMG failed
  MCL 3: pass, actual_nmg=4, iter=17, resid/resid0=3.62e-12
  MCL 4: pass, actual_nmg=4, iter=17, resid/resid0=3.83e-12
```

单层判断：

```text
- 单层 full-finest 的收敛不是 MCL 越深越好，但 MCL 太浅会直接失败。
- 对当前 s=5..9 full-finest，MCL=3 是最小稳定通用选择；MCL=2 对 s=7/s=9 不稳定，
  对 s=8 虽能过但 iter=32，明显不是稳健策略。
- 这说明 Stage 3.6 full-finest comparator 选 MCL=3 是合理的；后续单层准确性测试应固定
  MCL=3 或显式比较 MCL=3/4。
```

AMR 成本模型：

```text
s  full grid  AMR L0 grid  full/AMR solve cells  measured speedup  target gap to 3x
5  280x320    70x80        1.3636                0.179834          16.682
6  336x384    84x96        1.36956               0.176848          16.9637
7  392x448    98x112       1.37384               0.188732          15.8955
8  448x512    112x128      1.37706               0.169905          17.6569
9  504x576    126x144      1.37958               0.192967          15.5467
```

AMR 判断：

```text
- 当前 lower-left rectangular 三层 AMR layout 的 level-by-level solve cell count 只比
  full-finest 少约 1.36 到 1.38 倍。
- 这是理想 per-cell cost 模型下的乐观上限，已经远低于用户要求的大网格 3x speedup。
- 实测 speedup 只有 0.17 到 0.19，AMR 反而慢 5.2 到 5.9 倍。
- 因此，当前 rectangular level-by-level independent solve 框架无法通过低层优化达到
  >=3x speedup。继续只调 MCL/cache/小开销没有充分工程意义。
```

Stage 4 下一步建议：

```text
1. 单层准确性：
   - 固定 MCL=3，跑 static-rho grid refinement snapshot scan；
   - 用最高分辨率或 manufactured/reference solve 计算 phi/Er/Ez convergence；
   - 若单层误差异常，先修单层，不进入 AMR 准确性。

2. AMR 准确性：
   - 设计 exact artificial Dirichlet 对照或等价 reference-boundary test；
   - 分离 coarse solve error、coarse-to-fine interpolation error、artificial face error、
     EB BC/coarse-fine data error。

3. AMR 速度：
   - 若仍要求 >=3x speedup，必须重新设计 AMR 策略：
     either 减小 refined rectangular patches，or 进入 composite/coupled AMR Poisson solve 设计。
   - 当前 Ω1/Ω2 尺寸与 level-by-level 三次 solve 组合，在 cell-count 层面已经不具备 3x 速度优势。
```

Stage 4 第二批执行记录（2026-06-07）：

```text
新增/修改测试基础设施：
  Test/HallRZ/hallrz_stage4_single_level_reference_scan.py
    - 单层 full-finest static-rho reference scan；
    - 每个 s 输出 summary/snapshot/log；
    - 以最高分辨率 reference-s 的 snapshot 为数值参考；
    - 按 HallRZ Yee staggering 分别比较 phi_fp、Er_fp、Ez_fp；
    - 输出 CSV/Markdown，并计算相邻 s 的观测误差阶。

  Test/HallRZ/hallrz_stage3_6_amr_fullfinest_scan.py
    - 增加 --static-rho-test / --static-rho-mode / --static-rho-amp /
      --static-rho-lambda-r / --static-rho-lambda-z；
    - 用于隔离 AMR Poisson solve 本身误差，避免 PIC 粒子演化噪声混入判断。

生产代码修复：
  Source/EmbeddedBoundary/WarpXInitEB.cpp
  Source/FieldSolver/ElectrostaticSolvers/HallRZPoissonSolver.cpp

  修复内容：
    - 对已经通过 alignment 检查的 HallRZ lob/hib/out，在实际 EB2 初始化和
      HallRZ analytic nodal geometry/RHS/BC mask 使用前 snap 到当前 level cell-face 坐标；
    - runtime 几何分类容差使用 max(user hall_rz_align_tol, 1e-8*max(dx))；
    - ValidateGeometry 仍按原始用户输入做 alignment 检查，不改变 Python/User contract。

  修复原因：
    - s=7 full-finest static-rho reference scan 发现 r=hib 节点坐标为
      0.05000000000000001，严格 EB2/covered 归属把该节点当作 covered；
    - 直接后果是 phi/Er/Ez 在本应为 EB boundary 的位置被置零；
    - s=7 对 s=9 的 phi L2 rel 从正常 1e-5 量级跳到 3.24e-2，
      这是 WarpX HallRZ 几何 floating classification bug，不是 AMReX residual 收敛失败。
```

修复前单层 static-rho reference scan：

```text
command:
python Test/HallRZ/hallrz_stage4_single_level_reference_scan.py \
  --s-values 5 6 7 8 9 \
  --reference-s 9 \
  --steps 0 --chunk-steps 1 \
  --prefix hallrz_stage4_single_ref_static

outputs:
  Test/HallRZ/hallrz_stage4_single_ref_static.csv
  Test/HallRZ/hallrz_stage4_single_ref_static.md

key result:
  s=7 abnormal:
    phi L2 rel = 3.24145e-2
    Er  L2 rel = 1.30496e-1
    Ez  L2 rel = 1.09623e-1
    max error location at r=hib, z<out; target value was zero while reference was nonzero.
```

修复后单层 static-rho reference scan：

```text
commands:
cmake --build build -j 8 --target pip_install_nodeps

python Test/HallRZ/hallrz_stage4_single_level_reference_scan.py \
  --s-values 7 9 \
  --reference-s 9 \
  --steps 0 --chunk-steps 1 \
  --prefix hallrz_stage4_single_ref_static_snapfix_probe

python Test/HallRZ/hallrz_stage4_single_level_reference_scan.py \
  --s-values 5 6 7 8 9 \
  --reference-s 9 \
  --steps 0 --chunk-steps 1 \
  --prefix hallrz_stage4_single_ref_static_snapfix

outputs:
  Test/HallRZ/hallrz_stage4_single_ref_static_snapfix.csv
  Test/HallRZ/hallrz_stage4_single_ref_static_snapfix.md
```

修复后单层 reference 表：

```text
s  grid      NMG  iter  resid/resid0    phi L2 rel   Er L2 rel    Ez L2 rel
5  280x320   4    6     1.823e-12       5.47797e-05  7.45932e-03  1.38837e-03
6  336x384   4    10    8.808e-12       3.35128e-05  5.25062e-03  9.36242e-04
7  392x448   4    9     6.984e-12       1.92904e-05  3.25292e-03  5.66180e-04
8  448x512   4    10    6.605e-12       8.29217e-06  1.53922e-03  2.65327e-04
9  504x576   4    29    2.459e-12       0            0            0
```

单层 Stage 4 判断：

```text
- 修复后 s=5..8 对 s=9 的 phi/Er/Ez reference error 单调下降；
- s=7 的 face-classification 异常已消失；
- 单层 full-finest MCL=3 static-rho solve 可作为当前 AMR debug 的 reference path；
- 这一步只证明单层 reference path 可信，不证明三层 AMR 已通过。
```

修复后三层 AMR/full-finest 200 步扫描：

```text
command:
python Test/HallRZ/hallrz_stage3_6_amr_fullfinest_scan.py \
  --s-values 5 6 7 8 9 \
  --steps 200 --chunk-steps 20 \
  --full-finest-mcl 3 \
  --amr-mcls 0 1 2 \
  --prefix hallrz_stage4_snapfix_mcl3

outputs:
  Test/HallRZ/hallrz_stage4_snapfix_mcl3_amr_vs_fullfinest.csv
  Test/HallRZ/hallrz_stage4_snapfix_mcl3_amr_vs_fullfinest.md
```

修复后 200 步 AMR/full-finest 表：

```text
s  full grid  AMR L0 grid  full steady  AMR steady  speedup   phi L2 rel  Er L2 rel   Ez L2 rel
5  280x320    70x80        0.0412246    0.230291    0.179011  0.120925    0.00671425  0.00108241
6  336x384    84x96        0.0426624    0.240046    0.177726  0.110881    0.00543335  0.000871101
7  392x448    98x112       0.0439116    0.256419    0.171249  0.102984    0.00463884  0.000742757
8  448x512    112x128      0.0448625    0.265107    0.169224  0.0966322   0.00430372  0.000703177
9  504x576    126x144      0.0520412    0.270819    0.192162  0.0912700   0.00392066  0.000656744
```

修复后 static-rho 1 步 AMR/full-finest 隔离扫描：

```text
command:
python Test/HallRZ/hallrz_stage3_6_amr_fullfinest_scan.py \
  --s-values 5 6 7 8 9 \
  --steps 1 --chunk-steps 1 \
  --full-finest-mcl 3 \
  --amr-mcls 0 1 2 \
  --static-rho-test \
  --prefix hallrz_stage4_static_amr_vs_full_snapfix

outputs:
  Test/HallRZ/hallrz_stage4_static_amr_vs_full_snapfix_amr_vs_fullfinest.csv
  Test/HallRZ/hallrz_stage4_static_amr_vs_full_snapfix_amr_vs_fullfinest.md
```

static-rho AMR/full-finest 表：

```text
s  full grid  AMR L0 grid  full step  AMR step  speedup    phi L2 rel  Er L2 rel   Ez L2 rel
5  280x320    70x80        0.0101031  0.219339  0.0460618  0.124060    0.00814521  0.000916522
6  336x384    84x96        0.0113518  0.231352  0.0490672  0.113734    0.00678427  0.000763151
7  392x448    98x112       0.0118278  0.231061  0.0511892  0.105620    0.00581235  0.000653430
8  448x512    112x128      0.0131642  0.243814  0.0539926  0.0990937   0.00508496  0.000575127
9  504x576    126x144      0.0150234  0.245186  0.0612737  0.0935899   0.00451676  0.000503717
```

三层 AMR Stage 4 判断：

```text
- 修复后所有 AMR/full-finest JSON checks.pass=true，无 NaN/Inf，coarse/fine artificial
  Dirichlet boundary continuity diagnostic max_abs=0.0；
- 但是 200 步扫描中 AMR speedup_steady 只有 0.169 到 0.192，等价于 AMR 比
  full-finest 慢约 5.2 到 5.9 倍；
- static-rho 1 步隔离扫描仍给出 phi global L2 rel 约 0.094 到 0.124，
  level 2 phi L2 rel 约 0.107 到 0.141；
- 因此 AMR 的大 phi 误差不是 PIC 粒子演化噪声，而是当前
  level-by-level independent solve + coarse Dirichlet artificial boundary 框架本身的数值误差；
- coarse/fine boundary continuity 为 0 只说明人工边界节点被强制连续，不能证明 fine patch
  内部解接近 full-finest/composite reference；
- 当前 rectangular Ω1/Ω2 layout 的 solve-cell ratio 上限约 1.36 到 1.38，且实测还慢约 5 倍，
  不可能通过小修小补达到用户要求的大网格 >=3x speedup；
- Gate decision: 当前三层 AMR level-by-level 框架不通过 Stage 4 AMR accuracy/speed gate。
```

Stage 4 下一步必须讨论的方案问题：

```text
1. 若继续要求 AMR 与 full-finest/composite reference 误差可控，需要重新设计 composite/coupled
   AMR Poisson solve，或者至少加入 fine-to-coarse correction/flux matching，而不是只用
   one-way coarse Dirichlet。

2. 若继续要求 s=8/9 大网格 >=3x speedup，当前 Ω1/Ω2 refined patches 太大；即使 composite
   solve 正确，cell-count 层面也缺少 3x 空间。需要重新讨论 refined patch 尺寸、覆盖策略和
   真正的 composite solve 成本。

3. 进入下一轮代码前，必须先和用户确认：
   - AMR accuracy target 使用 static-rho/MMS/composite reference 的哪一种；
   - 是否放弃当前 level-by-level independent solve 作为正式方案；
   - 是否调整 Ω1/Ω2 几何，使 refined cell count 具备 >=3x speedup 的理论空间。
```

Stage 4 第三批信息收集（2026-06-07）：

```text
问题 1：单层 reference 表中 s=9 误差为 0 的含义

结论：
  - hallrz_stage4_single_ref_static_snapfix 的 reference-s 是 9；
  - 表中 s=9 行是 self-comparison，误差被脚本按定义置为 0；
  - 这不是相对解析解/真实解的误差为 0，也不是无穷/NaN 导致的异常。

finite check:
  file:
    Test/HallRZ/hallrz_stage4_single_ref_static_snapfix_s9.json
    Test/HallRZ/hallrz_stage4_single_ref_static_snapfix_s9.npz
    Test/HallRZ/hallrz_stage4_single_ref_static_snapfix_s9.log

  checks.pass=true
  phi_fp finite=true, nan=0, inf=0, min=0, max=0.1892845497, l2=0.1275524151
  Er_fp  finite=true, nan=0, inf=0, min=-5.905524435, max=5.947251944, l2=0.2735489056
  Ez_fp  finite=true, nan=0, inf=0, min=-9.614948001, max=1.144410028, l2=1.4877347914
  log reports N_NaN=0, N_Inf=0

extra non-self reference command:
  python Test/HallRZ/hallrz_stage4_single_level_reference_scan.py \
    --s-values 9 10 \
    --reference-s 10 \
    --steps 0 --chunk-steps 1 \
    --prefix hallrz_stage4_single_ref_s10_check

outputs:
  Test/HallRZ/hallrz_stage4_single_ref_s10_check.csv
  Test/HallRZ/hallrz_stage4_single_ref_s10_check.md

s=9 compared to s=10:
  phi L2 rel = 2.88270e-05
  Er  L2 rel = 1.28953e-03
  Ez  L2 rel = 2.23546e-04
  s=9 iter=31, NMG=4, resid/resid0=9.978e-12, N_NaN=0, N_Inf=0
```

```text
问题 2：AMR 表中 phi 相对误差比 E 大的原因

原 static-rho AMR/full-finest 表：
  s=9 global phi L2 rel = 9.35899e-02
  s=9 global Er  L2 rel = 4.51676e-03
  s=9 global Ez  L2 rel = 5.03717e-04

定位：
  - 原 compare_snapshots() 只按 AMR owned region 去掉 coarse/fine overlap；
  - 没有排除 HallRZ covered / non-computational nodes；
  - phi 是 nodal field，covered block 中的 phi 不应作为物理区域准确性指标；
  - E writeback 有 covered-side/非计算区域保护，E 的比较受该保护影响，不能和未加
    physical mask 的 phi global error 直接比较。

max-error evidence, s=9 static-rho AMR/full-finest:
  level 2 phi_fp max error at r=0.0501388889, z=0.0398611111,
  which is r>hib and z<out, i.e. high-r covered block.
  target phi = 0.1600886793, full reference phi = 0.0.

approx fluid-only recomputation, s=9:
  mask used for phi points:
    fluid if z >= out or (lob <= r <= hib and z <= out)
  This is an approximate physical-region mask for diagnosis only.

  level 0:
    phi L2 rel = 2.21873e-04
    Er  L2 rel = 7.84376e-04
    Ez  L2 rel = 4.30718e-04

  level 1:
    phi L2 rel = 2.15211e-04
    Er  L2 rel = 2.44020e-02
    Ez  L2 rel = 7.24488e-03

  level 2:
    phi L2 rel = 1.89541e-04
    Er  L2 rel = 3.23722e-03
    Ez  L2 rel = 3.78441e-04

判断：
  - 原表里 phi 比 E 大，主要是误差统计口径问题，不是物理区域内 phi 比 E 更差；
  - 后续 AMR accuracy 表必须区分:
      all-owned nodes,
      approximate/analytic fluid-only nodes,
      EB/cut-region附近节点，
      covered/non-computational nodes；
  - 原 all-owned phi global L2 rel 不能作为 AMR Poisson 物理准确性验收指标。
```

```text
问题 3：三层 AMR 每层求解时间与迭代数

临时诊断：
  - 在 HallRZPoissonSolver.cpp 中临时加入
    [SHULIU-CODEX-DIAG:amr-level-timing:20260607]；
  - 计时边界用 amrex::Gpu::streamSynchronize()；
  - 因此该 timing 会改变运行开销，只作为 per-level 定位数据；
  - 诊断跑完后已从 C++ 中移除，并重新执行 pip_install_nodeps 恢复正常版本。

build:
  cmake --build build -j 8 --target pip_install_nodeps

static-rho timing command:
  python Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
    --case hallrz_stage4_timing_s9_amr_static_1step \
    --nr 126 --nz 144 \
    --max-steps 1 --chunk-steps 1 \
    --summary-json Test/HallRZ/hallrz_stage4_timing_s9_amr_static_1step.json \
    --snapshot-npz Test/HallRZ/hallrz_stage4_timing_s9_amr_static_1step.npz \
    --warpx-max-grid-size 280 \
    --hall-rz-amr-enable \
    --hall-rz-amr-r1 0.060 --hall-rz-amr-z1 0.060 \
    --hall-rz-amr-r2 0.055 --hall-rz-amr-z2 0.050 \
    --hall-rz-max-coarsening-levels 0 1 2 \
    --eb-bc-mode robin \
    --particle-mode fixed \
    --static-rho-test

static-rho timing result:
  checks.pass=true, avg_step_wall_time_s=0.2566932300, init_wall_time_s=1.6690173560

  step=0:
    level 0: MCL=0, NMG=1, iter=22, setup=0.004786s, solve=0.771616s, total=0.776618s
    level 1: MCL=1, NMG=2, iter=4,  setup=0.002913s, solve=0.113884s, total=0.116959s
    level 2: MCL=2, NMG=2, iter=4,  setup=0.002508s, solve=0.127222s, total=0.130008s

  step=1:
    level 0: MCL=0, NMG=1, iter=0, setup=0.000487s, solve=0.000464s, total=0.001064s
    level 1: MCL=1, NMG=2, iter=4, setup=0.001219s, solve=0.112009s, total=0.113384s
    level 2: MCL=2, NMG=2, iter=4, setup=0.005151s, solve=0.123868s, total=0.129291s

50-step timing command:
  python Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
    --case hallrz_stage4_timing_s9_amr_fixed_50step_seq \
    --nr 126 --nz 144 \
    --max-steps 50 --chunk-steps 10 \
    --summary-json Test/HallRZ/hallrz_stage4_timing_s9_amr_fixed_50step_seq.json \
    --warpx-max-grid-size 280 \
    --hall-rz-amr-enable \
    --hall-rz-amr-r1 0.060 --hall-rz-amr-z1 0.060 \
    --hall-rz-amr-r2 0.055 --hall-rz-amr-z2 0.050 \
    --hall-rz-max-coarsening-levels 0 1 2 \
    --eb-bc-mode robin \
    --particle-mode fixed

50-step timing result:
  checks.pass=true, avg_step_wall_time_s=0.2729450823, init_wall_time_s=2.5626134030

  step>0 total HallRZ timed sum = 13.340406352 s

  level 0:
    count=50, MCL=0, NMG=1
    iter counts: iter=0 for 11 solves, iter=1 for 39 solves
    mean total=0.0275194 s, total sum=1.375970 s, contribution=10.31%

  level 1:
    count=50, MCL=1, NMG=2
    iter counts: iter=4 for 50 solves
    mean total=0.112322 s, total sum=5.616093 s, contribution=42.10%

  level 2:
    count=50, MCL=2, NMG=2
    iter counts: iter=4 for 50 solves
    mean total=0.126967 s, total sum=6.348344 s, contribution=47.59%

per-level timing conclusion:
  - PIC loop step>0 中最耗时的是 level 2；
  - level 2 比 level 1 略慢，二者合计约占 89.7% 的 HallRZ solve timed sum；
  - level 0 初始化 step=0 很贵（iter=22 或更高），但后续 warm-start 后通常 iter=0/1，
    只占 step>0 总耗时约 10.3%；
  - 当前三层 AMR 运行时主要成本来自 level 1/2 的重复独立 solve，而不是 level 0。

注意：
  - 曾经错误地并行启动 static 1-step 和 50-step 两个 GPU run，50-step 触发 GPU OOM；
  - 该失败数据已丢弃，随后顺序单独重跑 50-step 成功；
  - OOM 产生的 Backtrace.0 已删除。
```

Stage 4 第四批实施目标（2026-06-07）：

```text
目标：
  先修当前三层 AMR level-by-level path 的 fine-level initial guess 策略，验证性能瓶颈
  是否主要来自 level 1/2 每步被 coarse interpolation 全域覆盖，导致上一 PIC 步 fine phi
  热启动失效。

新增参数：
  warpx.hall_rz_amr_phi_init_weight

语义：
  对 lev>0 的 fine-level AMR solve，在 coarse level 已完成当前 PIC 步求解后，构造
  Interp(phi_coarse_current)。随后用

    phi_init = w * phi_fine_previous + (1-w) * Interp(phi_coarse_current)

  作为 fine-level MLMG initial guess。

  w = 1:
    纯上一 PIC 步 fine phi 热启动。

  w = 0:
    纯当前 coarse phi 插值，等价于当前旧行为的主要初始猜测口径。

  0 < w < 1:
    两个 predictor 的线性混合。

默认：
  w = 1。

硬约束：
  - 该参数只影响 initial guess，不改变 RHS、operator、physical BC、EB BC 或 AMReX solver。
  - AMR artificial high-r/high-z patch boundary 节点必须始终用当前 coarse interpolation
    强制刷新；这些节点不参与 w 混合，避免上一 PIC 步 fine phi 污染当前 coarse Dirichlet
    人工边界。
  - step=0、未来 regrid 后、或没有合法 fine history 时，必须强制 w_eff=0，使用当前
    coarse interpolation 初始化。
  - 当前 HallRZ AMR 首版仍不支持运行中 regrid；本轮实现只需要处理 step=0 fallback。
  - E-field covered/non-computational 区域保护不能修改。

测试矩阵：
  - s=9 static-rho short scan:
      w = 0, 0.25, 0.5, 0.75, 1
      max_steps = 1 或 2，chunk_steps = 1。
    主要看 step>0 的 level 1/2 iter 是否从 4 降到 0/1。

  - s=9 fixed-particle short scan:
      w = 0, 0.25, 0.5, 0.75, 1
      max_steps = 50，chunk_steps = 10。
    记录 avg_step_wall_time_s、steady step time、level 1/2 Final Iter、
    resid/resid0、NaN/Inf、coarse/fine artificial Dirichlet continuity。

验收：
  - checks.pass=true。
  - 无 NaN/Inf。
  - residual 维持原容差水平。
  - artificial Dirichlet continuity 不退化。
  - 若 w=1 在 static-rho/fixed-particle 下显著减少 level 1/2 iter 或 step time，则保留
    w=1 默认，并进入下一轮更完整测试。
  - 若 w=0 或混合权重在某些快速变化 PIC 工况下更优，保留参数作为用户可调策略。
```

AMR 网格设计建议补充：

```text
定义：
  full/AMR solve cell ratio =
    full finest 单层总 cell 数 /
    level-by-level AMR 各层实际 solve cell 数之和。

含义：
  这个 ratio 是只从 cell 数减少出发的乐观速度空间，不包含多层独立 solve、bottom solve、
  MPI communication、kernel launch、coarse/fine interpolation 或 EB geometry/BC 的额外开销。

示例：
  当前 s=9 lower-left three-level layout:
    full finest cells = 504 * 576 = 290304
    AMR level-by-level solved cells = 18144 + 47524 + 144076 = 209744
    full/AMR solve cell ratio ~= 1.38

判断：
  ratio ~= 1.38 意味着即使每个 cell 的 solve 成本完全相同，AMR 从 cell count 层面也很难
  带来大幅加速，更不具备 >=3x speedup 的空间。若用户希望三层 AMR 明显快于 full-finest，
  Python 端应尽量缩小 r1/z1/r2/z2，使 refined rectangular patches 只覆盖物理上必须高分辨率
  的区域。

限制：
  这是用户/benchmark 设计建议，不是 WarpX C++ 硬规则。WarpX 不根据该 ratio 自动拒绝、
  缩放或重写用户输入的 AMR boxes。
```

Stage 4 第四批执行记录（2026-06-07）：

```text
代码修改：
  Source/FieldSolver/ElectrostaticSolvers/LabFrameExplicitES.cpp
    - 新增 warpx.hall_rz_amr_phi_init_weight，默认 1.0，范围 [0,1]；
    - lev>0 coarse-to-fine phi 初始化改为：
        phi_init = w * phi_fine_previous + (1-w) * Interp(phi_coarse_current)
    - step=0 强制 w_eff=0；
    - AMR artificial high-r/high-z patch boundary 节点始终强制使用当前 coarse interpolation；
    - 不修改 RHS/operator/physical BC/EB BC/AMReX。

  Test/HallRZ/picmi_hall_rz_pic_benchmark.py
    - 新增 --hall-rz-amr-phi-init-weight；
    - 写入 pywarpx.warpx.hall_rz_amr_phi_init_weight；
    - JSON summary.solver 记录 hall_rz_amr_phi_init_weight。

  Test/HallRZ/hallrz_stage3_6_amr_fullfinest_scan.py
    - 新增 --amr-phi-init-weight；
    - AMR case 透传到 benchmark；
    - CSV 行记录 amr_phi_init_weight。

build/check:
  python -m py_compile \
    Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
    Test/HallRZ/hallrz_stage3_6_amr_fullfinest_scan.py

  git diff --check -- \
    Source/FieldSolver/ElectrostaticSolvers/LabFrameExplicitES.cpp \
    Test/HallRZ/picmi_hall_rz_pic_benchmark.py \
    Test/HallRZ/hallrz_stage3_6_amr_fullfinest_scan.py \
    HallRZ_EB_ROBIN_REFACTOR_PLAN_CN.md

  cmake --build build -j 8 --target pip_install_nodeps

结果：
  全部通过。
```

```text
s=9 static-rho 2-step weight scan:

w     checks.pass  avg_step_wall_time_s  chunk step1  chunk step2
0     true         0.247055              0.251419     0.242691
0.25  true         0.244777              0.248860     0.240695
0.5   true         0.212469              0.213192     0.211746
0.75  true         0.177578              0.178615     0.176540
1     true         0.0186577             0.0214693    0.0158461

判断：
  - static-rho 下 w 越接近 1 越快；
  - w=1 显著恢复 fine-level previous-phi 热启动；
  - 这支持“旧实现每步用 coarse interpolation 全域覆盖 fine phi，是 level 1/2 持续 4 iter
    的重要原因”。
```

```text
s=9 fixed-particle 50-step weight scan:

w     checks.pass  avg_step_wall_time_s  steady_s_per_step  step50 lev0/1/2 iter
0     true         0.262026              0.266138           1 / 4 / 4
0.25  true         0.261825              0.266178           1 / 4 / 4
0.5   true         0.261277              0.266545           1 / 4 / 4
0.75  true         0.230177              0.234982           1 / 3 / 4
1     true         0.109010              0.114145           1 / 1 / 1

step50 residual and finite checks:
  - all cases N_NaN=0, N_Inf=0；
  - w=1 step50:
      level 0 resid/resid0 = 5.654476426e-13
      level 1 resid/resid0 = 4.980815205e-12
      level 2 resid/resid0 = 6.963532633e-12

判断：
  - fixed-particle 下 w=1 把 level 1/2 step50 iter 从 4/4 降到 1/1；
  - AMR steady time 从 w=0 的 0.266138 s/step 降到 w=1 的 0.114145 s/step，
    约 2.33x 加速；
  - w=0.25/0.5 基本没有收益，w=0.75 有部分收益但仍明显慢于 w=1。
```

```text
s=9 AMR-vs-full snapshot compare with w=1:

command:
  python Test/HallRZ/hallrz_stage3_6_amr_fullfinest_scan.py \
    --s-values 9 \
    --steps 50 --chunk-steps 10 \
    --full-finest-mcl 3 \
    --amr-mcls 0 1 2 \
    --amr-phi-init-weight 1 \
    --prefix hallrz_stage4_phiinit_compare_w1

outputs:
  Test/HallRZ/hallrz_stage4_phiinit_compare_w1_amr_vs_fullfinest.csv
  Test/HallRZ/hallrz_stage4_phiinit_compare_w1_amr_vs_fullfinest.md

result:
  full steady = 0.0511742 s/step
  AMR  steady = 0.114899  s/step
  speedup     = 0.445382
  phi L2 rel  = 0.0912698
  Er  L2 rel  = 0.00390558
  Ez  L2 rel  = 0.000653321

  AMR checks.pass=true
  coarse/fine artificial Dirichlet continuity max_abs=0.0
  step50 AMR iter:
    level 0 = 1
    level 1 = 1
    level 2 = 1

判断：
  - w=1 显著恢复速度，但当前 level-by-level AMR 仍慢于 full-finest：
      AMR 比 full-finest 慢约 2.24x；
  - 这是相对旧 w=0/旧行为约 5.2x 慢的明显改善；
  - AMR accuracy 指标没有因 initial guess 策略变差，仍反映当前 level-by-level independent
    solve 框架的原有误差口径；
  - 下一步若继续追求 AMR 比 full-finest 快，需要继续处理：
      refined patch 过大导致 cell-count ratio 只有约 1.38；
      level-by-level independent solve 仍不是 composite/coupled solve。
```

## 8. Stage 4.5：三层 AMR cell-count ratio 大网格收益边界测试

Stage 4.5 目标：

```text
不修改 bottom solver，不修改 AMReX。

用同一物理域、同一 finest 分辨率下的 full-finest 单层 reference，对比三层 static
rectangular AMR，定量输出：

  1. 不同 full/AMR solve cell ratio 下的速度收益；
  2. 同一 case 下 AMR 相对 full-finest 的精度损失。

最终报告必须把 speedup 和 error 放在同一张主表里，方便直接判断三层 AMR 的工程收益是否
值得相应精度损失。
```

新增测试脚本：

```text
Test/HallRZ/hallrz_stage4_5_amr_ratio_scan.py
```

新增/扩展测试参数：

```text
Test/HallRZ/picmi_hall_rz_core_smoke.py
Test/HallRZ/picmi_hall_rz_pic_benchmark.py

新增 rmax/zmax 参数，仅用于 benchmark/test driver 扩展物理 plume 域。
默认仍保持原 HallRZ core smoke 物理域：
  rmax = 0.070
  zmax = 0.080

Stage 4.5 scan driver 还支持 custom full grid，用于严格指定非默认长宽比测试，例如
`2048x2048` 方形域。custom full grid 必须能被三层 AMR 的总 refinement ratio 4 整除。
```

为什么需要 `rmax/zmax` 扩展：

```text
在默认物理域 rmax=0.070, zmax=0.080 中，Level 2 patch 必须覆盖 channel:

  hib = 0.050
  out = 0.040

且三层 lower-left rectangular AMR 要求:

  hib < r2 < r1 < rmax
  out < z2 < z1 < zmax

因此默认域内 full/AMR solve cell ratio 的理论上限不到 2。若要测试 ratio~3/5/8，
必须让物理 plume 域相对 channel 变大，或让 channel 相对全域变小。Stage 4.5 脚本会对
每个 target ratio 先计算可行性；不可行时写入报告，不静默生成无效 patch。
```

Stage 4.5 ratio 定义：

```text
actual full/AMR solve cell ratio =
  full finest 单层 rho cell 数 /
  (AMR level 0 rho cell 数 + level 1 rho cell 数 + level 2 rho cell 数)

model ratio =
  1 / (1/16 + A1/4 + A2)

其中 A1/A2 分别是 Level 1/2 lower-left patch 占整个物理域的面积比例。
```

Stage 4.5 MCL 测试口径更新（2026-06-07）：

```text
后续 full-finest 与三层 AMR 的速度/精度对比，默认使用 deepest requested MCL：

  full-finest: requested MCL = 30
  AMR level 0/1/2: requested MCL = [30,30,30]

这里的 “deepest” 指 WarpX 不主动限制 max coarsening level，让 AMReX 根据网格、EB/RZ
operator 约束和内部 stop rule 决定实际 NMG levels。该设置不一定是每个 case 的最优速度点，
但相比过浅 MG 更接近稳定生产求解口径，也更适合作为跨网格设计的统一比较基准。

旧的 `MCL=0/1/2/3/5` 结果作为性能诊断历史保留；除非专门做 MCL sensitivity scan，后续
Stage 4.5 speedup/error 主表不再使用浅 MCL 作为默认比较口径。
```

Stage 4.5 主输出表必须至少包含：

```text
domain scale
full grid
target cell ratio
actual cell ratio
full steady s/step
AMR steady s/step
speedup = full steady / AMR steady
global phi L2 rel
global max(Er,Ez) L2 rel
Level2 phi L2 rel
Level2 max(Er,Ez) L2 rel
channel/EB phi L2 rel
channel/EB max(Er,Ez) L2 rel
pass/fail
```

误差统计口径：

```text
所有 full-vs-AMR field error 必须只统计 HallRZ fluid/computational region。

具体 mask:
  z >= out:
    plume 区域有效；
  z < out:
    只有 lob <= r <= hib 的 channel 区域有效。

不得把 covered solid / 非计算区域纳入 phi、Er、Ez 的 L2/Linf error。尤其是 phi 在 covered
区域没有物理误差意义；E-field 当前代码也有 covered/non-computational 区域不计算保护，误差
后处理必须尊重这个口径。
```

同时必须输出按 target ratio 聚合的设计判断表：

```text
target ratio
case count
actual ratio range
speedup range
phi error range
E error range
judgment
```

Stage 4.5 推荐测试命令：

```bash
# smoke: 小网格、少步数，验证脚本和误差输出
python Test/HallRZ/hallrz_stage4_5_amr_ratio_scan.py \
  --s-values 5 \
  --domain-scales 1 \
  --target-ratios 1.5 \
  --steps 2 --chunk-steps 1 \
  --max-grid-size 280 \
  --prefix hallrz_stage4_5_smoke

# 2048-class scan 示例：同一 finest spacing 下扩展 plume 域，覆盖 ratio~3/5/8 可行区
python Test/HallRZ/hallrz_stage4_5_amr_ratio_scan.py \
  --s-values 12 \
  --domain-scales 2 3 \
  --target-ratios 2 3 5 8 \
  --steps 200 --chunk-steps 20 \
  --max-grid-size 512 \
  --full-finest-mcl 30 \
  --amr-mcls 30 30 30 \
  --amr-phi-init-weight 1 \
  --prefix hallrz_stage4_5_large_ratio

# 严格 2048x2048 方形域示例：rmax=zmax=0.16，与 scale=2 的 z 向长度和 finest spacing 一致
python Test/HallRZ/hallrz_stage4_5_amr_ratio_scan.py \
  --custom-grid-label exact2048 \
  --custom-rmax 0.16 --custom-zmax 0.16 \
  --custom-nr-full 2048 --custom-nz-full 2048 \
  --target-ratios 2 3 5 8 \
  --steps 200 --chunk-steps 20 \
  --max-grid-size 2048 \
  --full-finest-mcl 30 \
  --amr-mcls 30 30 30 \
  --amr-phi-init-weight 1 \
  --prefix hallrz_stage4_5_exact2048_mcl5_amr312_mgs2048
```

Stage 4.5 验收：

```text
- full-finest reference checks.pass=true；
- AMR checks.pass=true；
- 无 NaN/Inf；
- AMR artificial Dirichlet continuity 不退化；
- 输出 CSV/Markdown 同时给 speedup 和 error；
- 不因为当前 case 的 profiler 中 bottom solver 占比高而修改 bottom solver；
- 如果 ratio 很小导致 AMR 慢，结论应写成 patch/cell-count 设计问题，而不是 solver bug；
- 如果 ratio 足够大仍慢，再单独报告 independent level-by-level solve fixed cost 证据。
```

Stage 4.5 严格 `2048x2048` 方形域历史结果（2026-06-07，浅 MCL 口径）：

```text
domain: rmax=zmax=0.16, full grid 2048x2048, L0 512x512
steps: 200, full MCL=5, AMR MCL=[3,1,2], max_grid_size=2048

target  actual ratio  full s/step  AMR s/step  speedup   phi rel      E rel max    status
2       1.98746       0.169954     0.300450    0.565666  9.006e-05   8.682e-04   pass
3       2.97226       0.169954     0.227558    0.746859  9.736e-05   2.795e-03   pass
5       4.94789       0.169954     0.213410    0.796375  1.047e-04   4.083e-03   pass
8       infeasible in this fixed square domain because Level 2 must cover channel.
```

该结果作为历史对照保留。后续默认以 deepest requested MCL 结果为准。

Stage 4.5 当前严格 `2048x2048` 方形域 deepest-MG 结果（2026-06-07）：

```text
domain: rmax=zmax=0.16, full grid 2048x2048, L0 512x512
steps: 200, full requested MCL=30, AMR requested MCL=[30,30,30], max_grid_size=2048
actual NMG:
  full: 10
  AMR target ratio 5: level0=8, level1=2, level2=2

target  actual ratio  full s/step  AMR s/step  speedup   phi rel      E rel max    status
2       1.98746       0.125586     0.275642    0.455611  9.008e-05   8.682e-04   pass
3       2.97226       0.125586     0.188330    0.666840  9.737e-05   2.795e-03   pass
5       4.94789       0.125586     0.179028    0.701485  1.047e-04   4.083e-03   pass
8       infeasible in this fixed square domain because Level 2 must cover channel.
```

结论：严格 `2048x2048`、该固定方形物理域内，deepest-MG 正确性通过，误差与浅 MCL 口径基本
一致；但 full-finest 单层在 deepest-MG 下也明显变快，因此当前 level-by-level 三层 AMR 仍未获得
速度收益。ratio~5 仍只有约 `0.70x`。这说明后续若要 AMR 体现速度优势，不能只依赖更深 MG；
仍需关注 independent three-solve fixed cost、solver/cache 生命周期和 patch/cell-count 设计。
