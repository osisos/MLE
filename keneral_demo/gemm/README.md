# GEMM kernel 示例

此目录集中保存 GEMM 的版本演进和基础指令实验。理论笔记见
[notes/GEMM.md](../../notes/GEMM.md)。以下命令均从仓库根目录执行，使用项目现有的 TIRx 环境。

## 目录与入口

| 文件 | 架构 / 范围 | 说明 |
| --- | --- | --- |
| `ada_gemm_v2.py` | Ada / SM89 | 当前 Ada 实现：`cp.async` 双缓冲、`ldmatrix`、`mma.sync`、寄存器整块写回；入口 `SM89_GEMM` |
| `gemm_v5.py` | Blackwell / B200 | 早期 TMA、TMEM、persistent CTA 示例；历史入口名保留为 `hgemm_v6` |
| `gemm_v6.py` | Blackwell / B200 | 双缓冲与 persistent tile scheduler；入口 `hgemm_v6` |
| `gemm_v7.py` | Blackwell / B200 | warp specialization 和 pipeline 封装；入口 `hgemm_v7` |
| `examples/ada_swizzle_ldmatrix_example.py` | SM8x 数据路径 | 独立验证 swizzle、`ldmatrix` 和 MMA 的 A fragment 布局 |
| `examples/mbarrier_sm8x.cu` | SM80 / SM86 / SM89 | 独立验证 `cp.async` 与 mbarrier 的多轮同步 |
| `drafts/ada_gemm_v1.py` | 历史草稿 | 保留早期推导；末尾 `mma_v2` 尚未完成，当前无法通过 Python 语法解析，不作为可运行入口 |

## 当前 Ada 算子

计算 `D = A @ B.T`：A 为 `(M, K)` FP16，B 为 `(N, K)` FP16，D 为 `(M, N)` FP32。
当前仅支持完整分块：M、N 分别为 128 的倍数，K 为 64 的倍数。
`SM_COUNT = 66` 是该示例的固定 CTA 数量；不同设备的性能调优需单独考虑。

从新路径导入并构造 kernel：

```python
from keneral_demo.gemm.ada_gemm_v2 import SM89_GEMM

kernel = SM89_GEMM(128, 128, 256)
```

构造函数返回 TIRx PrimFunc，不会自动运行 GPU kernel。编译或生成 CUDA 源码：

```python
import tvm

target = tvm.target.Target({"kind": "cuda", "arch": "sm_89"})
module = tvm.tirx.build(kernel, target=target, pipeline="tirx")
```

本地不启用 CUDA 的 TVM 可用于前端和源码生成检查；GPU 数值与性能验证需要启用 CUDA 的 TVM、CUDA 工具链和对应 NVIDIA GPU。

## 基础示例

CPU 地址映射检查和 TIRx lowering，无需 GPU：

```sh
.venv/bin/python keneral_demo/gemm/examples/ada_swizzle_ldmatrix_example.py
```

在 SM89 上额外验证 GPU 寄存器快照：

```sh
.venv/bin/python keneral_demo/gemm/examples/ada_swizzle_ldmatrix_example.py --run-cuda
```

编译、运行 mbarrier 示例：

```sh
nvcc -std=c++17 -O2 -arch=sm_89 keneral_demo/gemm/examples/mbarrier_sm8x.cu -o /tmp/mbarrier_sm8x
/tmp/mbarrier_sm8x
```

SM80、SM86 可替换相应的 `-arch`。B200 示例使用 TMA / TCGEN05 / TMEM，不应按 SM89 编译或运行。
