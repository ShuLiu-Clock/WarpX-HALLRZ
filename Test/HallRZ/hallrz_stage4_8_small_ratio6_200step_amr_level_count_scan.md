# HallRZ Stage 4.8 AMR Level-Count Scan

| grid | layout | actual ratio | steady s/step | speedup | phi L2 rel | E L2 rel max | phi Linf abs @ level/r/z | E Linf abs @ field/level/r/z | status |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| 224x256 | one | 1 | 0.00550159 | 1 | 0 | 0 | 0 @ L0/0/0 | 0 @ /L0/0/0 | pass |
| 224x256 | two | 3.56021 | 0.00824293 | 0.667431 | 0.000359209 | 0.0245924 | 0.000781975 @ L1/0.055/0.04 | 0.585518 @ Er_fp/L1/0.054375/0.04 | pass |
| 224x256 | three | 5.50452 | 0.0111868 | 0.491791 | 0.00108791 | 0.013519 | 0.000526323 @ L1/0.045/0.09 | 0.372362 @ Er_fp/L1/0.10875/0.04 | pass |
| 336x384 | one | 1 | 0.00555556 | 1 | 0 | 0 | 0 @ L0/0/0 | 0 @ /L0/0/0 | pass |
| 336x384 | two | 3.59783 | 0.00754991 | 0.735844 | 0.000225433 | 0.0194943 | 0.000609952 @ L1/0.0533333/0.04 | 0.622514 @ Er_fp/L1/0.0529167/0.04 | pass |
| 336x384 | three | 0 | 0 | 0 | 0 | 0 | 0 @ L/0/0 | 0 @ /L/0/0 | failed_returncode_6: /home/shuliu/work/cuda_warpx_202604/WarpX/Test/HallRZ/hallrz_stage4_8_small_ratio6_200step_336x384_three_r6p0.log |

Fastest passing layout by grid:

| grid | fastest layout | steady s/step | speedup vs full | note |
|---|---|---:|---:|---|
| 224x256 | one | 0.00550159 | 1 | full fastest |
| 336x384 | one | 0.00555556 | 1 | full fastest |
