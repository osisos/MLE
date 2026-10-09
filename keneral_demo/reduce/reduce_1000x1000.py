"""练习目标：用 TIRx 实现 1000 × 1000 矩阵的逐行最大值归约。

输入：X，形状 (1000, 1000)，行优先连续存储，dtype 为 float16。
输出：Y，形状 (1000,)，dtype 为 float16，只返回最大值，不返回索引。
计算：Y[i] = max(X[i, j] for j in range(1000))，即沿 axis=1 归约。

验收要求：
- 与 NumPy 参考结果 X.max(axis=1) 一致。
- 覆盖全负数、重复最大值，以及最大值位于每行最后一列的情况。
- 正确处理 1000 列的边界，不能越界读取，也不能遗漏末尾元素。
- 本练习先限定输入为有限值，暂不考虑 NaN / Inf 的语义。

沿用现有练习的 FP16 数据类型和 CUDA SM89 目标；kernel 实现自行完成。
依赖：NumPy，以及包含 tvm.tirx 的 TVM 构建。
GPU 运行还需要启用 CUDA 的 TVM 和对应的 NVIDIA GPU。
"""

import numpy as np
import tvm
from tvm.tirx import script as T

N = 1000
DTYPE = "float16"
TARGET = {"kind": "cuda", "arch": "sm_89"}

BLOCK_CNT = 125
WARP_PER_CTA = 8


# 运气很好，你要求1000x1000的算子，1000 x 2 % 16 = 0，需要输入16对齐
def SM89_1000x1000_REDUCE_LINE_MAX():

    @T.prim_func
    def kernel(
        X: T.Buffer(
            (
                N,
                N,
            ),
            "float16",
        ),
        OUT: T.Buffer((N,), "float16"),
    ):
        bx = T.cta_id([BLOCK_CNT])
        warp_id = T.warp_id([WARP_PER_CTA])
        lane_id = T.lane_id([32])

        num_buffer = T.alloc_local((4,), "float32")
        num = num_buffer.view("float16")

        #
        def load_gmem_to_reg(ptr):
            # float4 向量化加载
            T.ptx.ld(
                ptr,
                "uint32",
                "b32",
                dst=num.ptr_to([0]),
                space="global",
                vec="v4",
            )

        rounds = N // 256  # 8 * 32 = warp 每一轮可以处理的元素数

        tail_start = N - (N % 256)

        for r in range(rounds):




    return kernel
