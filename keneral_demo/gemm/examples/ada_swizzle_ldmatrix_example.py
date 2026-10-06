"""SM8x 学习示例：A[16,64] global -> swizzled smem -> MMA 的 A fragment。

从仓库根目录运行（使用当前项目的 TIRx 版本）：
    .venv/bin/python keneral_demo/gemm/examples/ada_swizzle_ldmatrix_example.py
    .venv/bin/python keneral_demo/gemm/examples/ada_swizzle_ldmatrix_example.py --run-cuda

默认在 CPU 上验证地址映射，并解析、lower 真正的 TIRx kernel；不需要 GPU。
--run-cuda 额外编译并运行 kernel，逐元素检查 GPU 导出的寄存器快照。

只展开 A 的数据路径，便于看清地址。这里没有执行完整 GEMM。
RegDump[k_step, lane, e] 保存的是 FP16 数值，不是地址。
它的 e 顺序就是 mma.m16n8k16.row.col 所需的 A 操作数顺序：
    a_reg0 = (e0,e1), a_reg1 = (e2,e3),
    a_reg2 = (e4,e5), a_reg3 = (e6,e7)。

PTX 依据：
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float
"""

import argparse

import numpy as np
import tvm
from tvm.tirx import script as T


def swizzled_offset(m, k):
    """返回物理 FP16 下标（不是字节地址），仅适用于本例每行 64 个 FP16。

    一行分成 8 个 16B 片段，每片段 8 个 FP16：
        原片段号 = k // 8
        物理片段号 = (k // 8) XOR (m % 8)
        片段内位置 = k % 8，保持不变

    合并后就是下面的 XOR。它等价于本例形状的
    mma_shared_layout("float16", SWIZZLE_128B_ATOM, (16,64))。
    例如 A[1,0:8] 存到 Smem[72:80]，而不是 Smem[64:72]。
    """
    return m * 64 + (k ^ ((m % 8) * 8))


@T.prim_func
def load_a_fragments(
    A: T.Buffer((16, 64), "float16"),
    RegDump: T.Buffer((4, 32, 8), "float16"),
):
    T.func_attr({"global_symbol": "load_a_fragments", "tirx.noalias": True})
    T.device_entry()
    bx = T.cta_id([1])
    warp = T.warp_id([1])
    lane = T.lane_id([32])

    # 用一维物理 buffer，把每一次 swizzle 寻址显式写出来。
    # 不再给 Smem 附加 swizzled layout，否则会把地址变换做两遍。
    pool = T.SMEMPool()
    Smem = pool.alloc((16 * 64,), "float16", align=128)
    pool.commit()

    # 每个 lane 独立持有 8 个 FP16，供 4 个 b32 操作数使用。
    # 这里分配的是线程私有存储；不再需要 @laneid 的分布式 layout。
    Areg = T.alloc_local((8,), "float16", align=4)

    # 1. global -> smem(swizzle)
    # 一个 lane 每次 cp.async 搬 16B = 8 个连续 FP16。
    # 32 lanes * 4 轮 * 8 FP16 = 16*64，恰好搬完整个 A tile。
    for i in T.unroll(4):
        chunk = i * 32 + lane
        m = chunk // 8
        k = (chunk % 8) * 8
        # global 地址沿逻辑行连续，只有 shared 目标地址进行 swizzle。
        # swizzle 保留片段内部顺序，因此一次 16B copy 不会写错元素。
        T.ptx.cp_async(
            Smem.ptr_to([swizzled_offset(m, k)]),
            A.ptr_to([m, k]),
            16,  # cp.async 的大小单位是字节
        )

    T.ptx.cp_async.commit_group()
    T.ptx.cp_async.wait_group(0)
    # wait_group 等待各线程自己的异步复制；CTA barrier 再让所有线程
    # 会合，确保随后 ldmatrix 读取别的线程写入的数据也已就绪。
    T.cuda.cta_sync()

    # 2. smem -> reg
    # K=64 切成四个 K=16 fragment。每次 x4 读取四个 8x8：
    #
    #                  K: k0..k0+7    k0+8..k0+15
    #   M: 0..7             X0             X2
    #   M: 8..15            X1             X3
    #
    # X0/X1/X2/X3 的顺序必须对应 MMA 的 a_reg0/1/2/3。
    for ks in T.unroll(4):
        k0 = ks * 16
        matrix = lane // 8
        row_in_matrix = lane % 8

        # lane 0..7   给 X0 提供八个行地址；lane 8..15 给 X1 提供。
        # lane 16..23 给 X2 提供；lane 24..31 给 X3 提供。
        # 注意：提供行地址的 lane，不等于最终接收整行数据的 lane！
        logical_m = (matrix % 2) * 8 + row_in_matrix
        logical_k = k0 + (matrix // 2) * 8

        # 找到逻辑片段的实际存放位置。无需先把 smem 排回普通行主序。
        T.ptx.ldmatrix(
            False,  # 不带 .trans
            4,      # .x4：四个 8x8 FP16 小矩阵，warp 合计读取 512B
            ".b16",
            Smem.ptr_to([swizzled_offset(logical_m, logical_k)]),
            Areg.ptr_to([0]),  # X0 -> e0,e1，即 a_reg0
            Areg.ptr_to([2]),  # X1 -> e2,e3，即 a_reg1
            Areg.ptr_to([4]),  # X2 -> e4,e5，即 a_reg2
            Areg.ptr_to([6]),  # X3 -> e6,e7，即 a_reg3
        )

        # 到这里 Areg 已经满足 MMA 的 A fragment 分布，没有额外重排。
        # 令 g=lane//4，t=lane%4，每个 lane 得到的实际数值来自：
        # e0,e1: A[g,   k0+2*t : k0+2*t+2]
        # e2,e3: A[g+8, k0+2*t : k0+2*t+2]
        # e4,e5: A[g,   k0+8+2*t : k0+8+2*t+2]
        # e6,e7: A[g+8, k0+8+2*t : k0+8+2*t+2]
        # 它不是原矩阵的行主序，而是 MMA 规定的 fragment 顺序。
        # 实际 GEMM 可在这里把四个寄存器交给 mma；本例导出便于观察。
        for e in T.unroll(8):
            RegDump[ks, lane, e] = Areg[e]


def expected_fragments(a):
    """直接按 PTX 的 MMA 坐标规范生成期望值，不经过 swizzle。"""
    expected = np.empty((4, 32, 8), dtype=np.float16)
    for ks in range(4):
        for lane in range(32):
            for e in range(8):
                m = lane // 4 + 8 * ((e // 2) % 2)
                k = ks * 16 + 2 * (lane % 4) + e % 2 + 8 * (e // 4)
                expected[ks, lane, e] = a[m, k]
    return expected


def verify_on_cpu(a):
    """模拟物理写入和 ldmatrix 分发，并与独立的 MMA 坐标规范比较。"""
    from tvm.backend.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout

    smem = np.empty(16 * 64, dtype=np.float16)
    layout = mma_shared_layout("float16", SwizzleMode.SWIZZLE_128B_ATOM, (16, 64))
    analyzer = tvm.arith.Analyzer()
    offsets = set()
    for m in range(16):
        for k in range(64):
            offset = swizzled_offset(m, k)
            offsets.add(offset)
            actual = layout.apply(m, k, shape=(16, 64))["m"]
            assert int(analyzer.simplify(actual)) == offset
            smem[offset] = a[m, k]
    assert offsets == set(range(16 * 64))  # 置换没有覆盖、遗漏或越界

    result = np.empty((4, 32, 8), dtype=np.float16)
    for ks in range(4):
        for matrix in range(4):
            banks = []
            for row in range(8):
                m = (matrix % 2) * 8 + row
                k = ks * 16 + (matrix // 2) * 8
                start = swizzled_offset(m, k)
                assert (start * 2) % 16 == 0
                # 每个 16B 片段占四个 bank；八行合起来恰好覆盖 32 banks。
                banks.extend(((start * 2 + b) // 4) % 32 for b in range(0, 16, 4))
                for p in range(8):
                    assert swizzled_offset(m, k + p) == start + p
                    # 不带 .trans 的 ldmatrix：每四个接收 lane 分一行。
                    receiver = row * 4 + p // 2
                    result[ks, receiver, matrix * 2 + p % 2] = smem[start + p]
            assert sorted(banks) == list(range(32))
    np.testing.assert_array_equal(result, expected_fragments(a))
    return result


def lower_kernel():
    """解析并执行 TIRx lowering；这是编译前端验证，不是 GPU 执行。"""
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_89"})
    mod = tvm.IRModule({"load_a_fragments": load_a_fragments})
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    pipeline, _, _ = tvm.tirx.get_tir_pipeline("tirx")
    return pipeline(mod)


def run_cuda(a):
    if not tvm.runtime.enabled("cuda") or not tvm.cuda(0).exist:
        raise RuntimeError("--run-cuda 需要启用 CUDA 的 TVM、CUDA 工具链和 NVIDIA GPU")
    device = tvm.cuda(0)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_89"})
    module = tvm.tirx.build(load_a_fragments, target=target, pipeline="tirx")
    a_device = tvm.runtime.tensor(a, device=device)
    output = tvm.runtime.empty((4, 32, 8), "float16", device=device)
    module["load_a_fragments"](a_device, output)
    device.sync()
    np.testing.assert_array_equal(output.numpy(), expected_fragments(a))
    print("PASS: GPU 寄存器快照全部符合 MMA 的 A fragment 布局")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-cuda", action="store_true", help="额外在 SM89 上编译并运行")
    args = parser.parse_args()
    # 0..1023 均可被 FP16 精确表示。数值就是逻辑 m*64+k，方便追踪。
    a = np.arange(16 * 64, dtype=np.float16).reshape(16, 64)
    result = verify_on_cpu(a)
    lower_kernel()
    print("PASS: CPU 验证 swizzle、16B 连续性、bank 分布及全部 1024 个 fragment 元素")
    print("PASS: TIRx kernel 已解析并 lower（未代表 GPU 验证）")
    print("A[m,k] = m*64+k；第一次 ldmatrix 后：")
    for lane in (0, 1, 4, 31):
        print(f"  lane {lane:2d}: {result[0, lane].astype(int).tolist()}")
    if args.run_cuda:
        run_cuda(a)
