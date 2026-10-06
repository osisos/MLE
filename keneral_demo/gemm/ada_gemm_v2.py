"""SM89 GEMM：cp.async 双缓冲、mma.sync 累加与 CTA 整块写回。"""

from typing import cast

import tvm
from tvm.ir import Var
from tvm.tirx import script as T
from tvm.tirx.layout import TileLayout, S, laneid, warpid
from tvm.backend.cuda.tile_primitive.tma_utils import mma_shared_layout, SwizzleMode
from tvm.backend.cuda.lang.tile_scheduler import ClusterPersistentScheduler2D
from tvm.tirx.script import tile as Tx

SM_COUNT = 66


def SM89_GEMM(M, N, K):
    """构造 D = A @ B.T 的 kernel；M/N 按 128、K 按 64 整除。"""
    a_type, b_type = tvm.DataType("float16"), tvm.DataType("float16")
    d_type = tvm.DataType("float32")

    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    assert M % BLK_M == 0, "M must be divisible by BLK_M (128)"
    assert N % BLK_N == 0, "N must be divisible by BLK_N (128)"
    assert K % BLK_K == 0, "K must be divisible by BLK_K (64)"
    K_STPS = K // BLK_K
    PIPE_DEPTH = 2

    WG_NUMBER = 1

    # 寄存器 累计 结果
    acc_per_thread = (BLK_M * BLK_N) // (32 * 4)

    A_layout = mma_shared_layout(
        a_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        [PIPE_DEPTH, BLK_M, BLK_K],
    )

    B_layout = mma_shared_layout(
        b_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        [PIPE_DEPTH, BLK_N, BLK_K],
    )

    # 注意这个layout这里没有声明原始数据是 [128 x 128]
    C_layout = TileLayout(
        S[(8, 2, 8, 4, 4, 4, 2) : (16, 2, 4 @ laneid, 4, 1 @ warpid, 1 @ laneid, 1)]
    )

    @T.prim_func
    def kernel(
        A: T.Buffer((M, K), a_type),  # pyright: ignore[reportInvalidTypeForm]
        B: T.Buffer((N, K), b_type),  # pyright: ignore[reportInvalidTypeForm]
        D: T.Buffer((M, N), d_type),  # pyright: ignore[reportInvalidTypeForm]
    ):
        T.device_entry()

        bx = T.cta_id([SM_COUNT])
        wg_id = T.warpgroup_id([1])
        warp_id = cast(Var, T.warp_id_in_wg([4]))
        lane_id = cast(Var, T.lane_id([32]))

        # tile scheduler 整个算子维度
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // BLK_M,
            num_n_tiles=N // BLK_N,
            l2_group_size=8,
            num_clusters=SM_COUNT,
        )
        tile_scheduler.init(bx)

        # 分配 smem 内存
        pool = T.SMEMPool()

        # mbarrier 设置
        smem_ready_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        smem_free_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        acc_ready_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        acc_free_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)

        Asmem = pool.alloc([PIPE_DEPTH, BLK_M, BLK_K], a_type, layout=A_layout)
        Bsmem = pool.alloc([PIPE_DEPTH, BLK_N, BLK_K], b_type, layout=B_layout)

        pool.commit()

        # mbarrier 初始化
        if warp_id == 0 and lane_id == 0:
            for s in range(PIPE_DEPTH):
                T.ptx.mbarrier.init(smem_ready_bar.ptr_to([s]), 128)
                T.ptx.mbarrier.init(smem_free_bar.ptr_to([s]), 128)
                T.ptx.mbarrier.init(acc_ready_bar.ptr_to([s]), 128)
                T.ptx.mbarrier.init(acc_free_bar.ptr_to([s]), 128)

        T.cuda.cta_sync()  # 这一句有必要吗？

        phase = T.alloc_local((PIPE_DEPTH,), "uint32")
        for s in range(PIPE_DEPTH):
            phase[s] = T.uint32(0)

        # C Reg 需要分配
        Creg = T.alloc_local(
            (BLK_M, BLK_N), acc_type, layout=C_layout
        )  # 分配所有的寄存器，# CTA? 为什么执行维度就是CTA

        Creg_local = (
            Creg.local()
        )  # local？去掉所有的线程轴，然后打平成一维排列？这个结果应该是包含逻辑

        Creg_thread = Creg.local(16, 8, layout=C_layout.storage())
        # C_layout.storage() 去掉所有的warp\lane_id 分量，代表仅看线程内的分布
        # local(16, 8 )

        """
            使用 cp_async
            BLK_M, BLK_K, BLK_N = 128, 64, 128
            每一个矩阵都是行主序，
            share memory 的 bank 宽度为 128B，swizzle的对齐宽度是16B
        """

        # 搬运 global memory -> share memory
        t_A = T.meta_var(
            (BLK_M * BLK_K * 2) // (4 * 16 * 32)
        )  # 32 * 4 (thread) * 16。A的搬运轮次

        t_B = T.meta_var((BLK_N * BLK_K * 2) // (4 * 16 * 32))
        warp_row = T.meta_var(
            (32 * 16) // (BLK_K * 2)
        )  # 32个线程，每个线程16B，每个线程搬运 BLK * 2B，这里有一个问题，为什么
        chunks_per_row = T.meta_var((BLK_K * 2) // 16)  # chunks 每行
        elems_per_copy = T.meta_var(16 // 2)  # 16B / sizeof(float16)

        @T.inline
        def clean_acc_reg():
            for i in range(acc_per_thread):
                Creg_local[i] = T.float32(0.0)

        # 按照外积累加
        @T.inline
        def load_gmem_to_smem(stage, m_offset, n_offset, k_offset):
            # 每一行给8个thread搬运，每一轮(32 * 4)
            for t in range(t_A):
                # 行的 offset
                row = (
                    t * (BLK_M // t_A)  # 轮次产生的行数offset
                    + (warp_id) * warp_row  # warp 产生的offset
                    + (lane_id // chunks_per_row)  # thread id 产生的 offset
                )

                # chunk 偏移
                col = (lane_id % chunks_per_row) * elems_per_copy

                # 接下来真正执行搬运，因为swizzle位置信息已经被写入 Aseme 的布局了，因此没必要手动计算swizzle了
                T.ptx.cp_async(
                    Asmem.ptr_to([stage, row, col]),
                    A.ptr_to([m_offset + row, k_offset + col]),
                    16,  # 每线程复制的字节大小
                )

            for t in range(t_B):
                # 同A的搬运
                row = (
                    t * (BLK_N // t_B)
                    + (warp_id) * warp_row
                    + (lane_id // chunks_per_row)
                )

                col = (lane_id % chunks_per_row) * elems_per_copy

                T.ptx.cp_async(
                    Bsmem.ptr_to([stage, row, col]),
                    B.ptr_to([n_offset + row, k_offset + col]),
                    16,
                )

        """
            MMA 执行过程，
            每个warp负责一个加载计算
        """
        MMA_M = T.meta_var(16)
        MMA_K = T.meta_var(16)
        MMA_N = T.meta_var(8)

        # 固定尺寸保留为元编程常量，避免生成运行时标量。
        mma_m_tiles = T.meta_var(BLK_M // MMA_M)  # 8
        mma_n_tiles = T.meta_var(BLK_N // MMA_N)  # 16
        k_steps = T.meta_var(BLK_K // MMA_K)  # 4

        mma_counts = T.meta_var(mma_m_tiles * mma_n_tiles)  # 总的输出矩阵的块数
        mma_rounds = T.meta_var(mma_counts // 4)  # 每个warp负责的输出块
        acc_pre_thread_round = T.meta_var((MMA_M * MMA_N) // 32)

        # 计算结果写回global memory中
        @T.inline
        def write_back(m_offset, n_offset):
            Tx.cta.copy(
                D[m_offset : m_offset + BLK_M, n_offset : n_offset + BLK_N], Creg[:, :]
            )

        @T.inline
        def mma_v1(stage):

            for r in range(mma_rounds):

                tile_idx: T.let = r * 4 + warp_id

                # 计算各个线程负责的输出块
                m_tile_idx: T.let = tile_idx // mma_n_tiles
                n_tile_idx: T.let = tile_idx % mma_n_tiles

                # 计算累加的C线程位置
                c_ptrs = [
                    Creg_thread.ptr_to(
                        [
                            m_tile_idx * 2 + i // 2,
                            (n_tile_idx // 4) * 2 + i % 2,
                        ]
                    )
                    for i in range(acc_pre_thread_round)
                ]

                # 以下是搬运

                for k in range(k_steps):
                    # ld matrix，这个API 需要每个thread单独计算本地线程需要搬运的位置

                    # ld A, A tile = [16, 16]，然而 ldmatrix最多支持 m8n8， x0,1,2,3 = [左上，左下，右上，右下]
                    # 接下来是计算每一个lane应该搬运的起始元素。（位置）
                    a_r_offset = lane_id % 16
                    a_c_offset = (lane_id // 16) * 8

                    A_reg = T.alloc_local((4,), "uint32")

                    T.ptx.ldmatrix(
                        False,  # 是否转置
                        4,  # 4x
                        ".b16",  # 类型
                        Asmem.ptr_to(
                            [
                                stage,
                                m_tile_idx * MMA_M + a_r_offset,  # row
                                k * MMA_K + a_c_offset,
                            ]
                        ),
                        *[A_reg.ptr_to([i]) for i in range(4)]
                    )

                    # --- 接下来是 ld B ---
                    # B = [8, 16] row master
                    b_r_offset = lane_id % 8
                    b_c_offset = (lane_id // 8) * 8

                    B_reg = T.alloc_local((2,), "uint32")

                    T.ptx.ldmatrix(
                        False,
                        2,
                        ".b16",
                        Bsmem.ptr_to(
                            [
                                stage,
                                n_tile_idx * MMA_N + b_r_offset,
                                k * MMA_K + b_c_offset,
                            ]
                        ),
                        *[B_reg.ptr_to([i]) for i in range(2)]
                    )

                    # 真实的执行MMA的过程
                    T.ptx.mma(
                        "m16n8k16",
                        "row",
                        "col",
                        "float32",
                        "float16",
                        "float16",
                        "float32",
                        c_ptrs,
                        [A_reg.ptr_to([i]) for i in range(4)],  # A
                        [B_reg.ptr_to([i]) for i in range(2)],  # B
                        c_ptrs,
                    )

        @T.inline
        def mma_v2(stage):
            # 进一步提高访存的局部性
            row_warp_rounds = T.meta_var(mma_rounds // mma_m_tiles)  # 每行 4 轮
            acc_per_res_tile = T.meta_var((MMA_M * MMA_N) // 32)

            for m_tile_idx in range(mma_m_tiles):
                for k in range(k_steps):

                    # 加载 A[m, k] -> reg
                    a_r_offset = lane_id % 16
                    a_c_offset = (lane_id // 16) * 8

                    A_reg = T.alloc_local((4,), "uint32")
                    T.ptx.ldmatrix(
                        False,
                        4,
                        ".b16",
                        Asmem.ptr_to(
                            [
                                stage,
                                m_tile_idx * MMA_M + a_r_offset,
                                k * MMA_K + a_c_offset,
                            ]
                        ),
                        *[A_reg.ptr_to([i]) for i in range(4)]
                    )

                    # 加载 B
                    # 这个B寄存器如何复用？
                    B_reg = T.alloc_local((2,), "uint32")

                    for n_in in range(row_warp_rounds):  # [0, 4]
                        n_tile_idx = n_in * 4 + warp_id

                        b_r_offset = lane_id % 8
                        b_c_offset = (lane_id // 8) * 8

                        B_reg = T.alloc_local((2,), "uint32")

                        T.ptx.ldmatrix(
                            False,
                            2,
                            ".b16",
                            Bsmem.ptr_to(
                                [
                                    stage,
                                    n_tile_idx * MMA_N + b_r_offset,
                                    k * MMA_K + b_c_offset,
                                ]
                            ),
                            *[B_reg.ptr_to([i]) for i in range(2)]
                        )

                        # 累加器结果位置? 我觉得这个很难写？能否优化编译器，改成直接 c layout 计算出应该在的位置，还是这意味着某种实现，根源上就不可以？
                        c_ptrs = [
                            Creg_thread.ptr_to(
                                [
                                    m_tile_idx * 2 + i // 2,
                                    (n_tile_idx // 4) * 2 + i % 2,
                                ]
                            )
                            for i in range(acc_per_res_tile)
                        ]

                        T.ptx.mma(
                            "m16n8k16",
                            "row",
                            "col",
                            "float32",
                            "float16",
                            "float16",
                            "float32",
                            c_ptrs,
                            [A_reg.ptr_to([i]) for i in range(4)],
                            [B_reg.ptr_to([i]) for i in range(2)],
                            c_ptrs,
                        )

        # 接下来把 完整流程串起来。需要注意 t.barrier 的同步原语的使用。
        # TODO； 这里有一个问题，sm9x 和 sm8x 的对应的barrier 如何使用？
        # 注意每个指令的执行范围，这是一个二阶段流水。

        # 阻塞等待，确定轮次完成后才继续执行。
        @T.inline
        def wait_phase(bar, p):  # bar = 地址， p = 数值
            while T.ptx.mbarrier_test_wait_parity(bar, p) == 0:  # 非阻塞等待？
                T.evaluate(0)

        while tile_scheduler.valid():
            m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
            n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

            # 清理 acc register
            clean_acc_reg()

            # 预填充，为什么要预填充？
            for s in range(PIPE_DEPTH):
                if s < K_STPS:
                    # load
                    load_gmem_to_smem(s, m_st, n_st, s * BLK_K)

                    # 异步，登记复制完成事件：此前发出的复制，完成后做一次到达。此时不一定复制完成
                    T.ptx.cp_async.mbarrier.arrive(smem_ready_bar.ptr_to([s]))  # +1 2

                    # 意味着本轮已经登记提交完毕？
                    T.ptx.mbarrier.arrive(smem_ready_bar.ptr_to([s]))  # -1 1

                    # 以上两条指令不能交换，真实复制完成 - 1

            for ki in range(K_STPS):
                s: T.let = ki % PIPE_DEPTH  # 计算 归属 pipe phase

                # 等待 golbal -> share mem
                wait_phase(smem_ready_bar.ptr_to([s]), phase[s])

                # 2. ldmatrix & mma
                mma_v1(s)  # 同步还是异步？线程可以做其他的事情吗？

                # 3. 报告本线程执行完毕
                T.ptx.mbarrier.arrive(smem_free_bar.ptr_to([s]))

                wait_phase(smem_free_bar.ptr_to([s]), phase[s])

                # load 下一个阶段
                phase[s] = phase[s] ^ T.uint32(1)

                if ki + PIPE_DEPTH < K_STPS:
                    load_gmem_to_smem(s, m_st, n_st, (ki + PIPE_DEPTH) * BLK_K)
                    T.ptx.cp_async.mbarrier.arrive(
                        smem_ready_bar.ptr_to([s])
                    )  # +1 res = 2
                    T.ptx.mbarrier.arrive(smem_ready_bar.ptr_to([s]))  # -1 res = 1

            # 接一下 Creg 的结果返回
            write_back(m_st, n_st)
            tile_scheduler.next_tile()

    return kernel
