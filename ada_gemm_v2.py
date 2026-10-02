from typing import cast

import tvm
from tvm.ir import Var
from tvm.tirx import script as T
from tvm.tirx.layout import TileLayout, S, laneid, warpid
from tvm.backend.cuda.tile_primitive.tma_utils import mma_shared_layout, SwizzleMode
from tvm.backend.cuda.lang.tile_scheduler import ClusterPersistentScheduler2D

SM_COUNT = 66


# c = a *b + d
def SM89_GEMM(M, N, K):
    a_type, b_type = tvm.DataType("float16"), tvm.DataType("float16")
    d_type = tvm.DataType("float32")

    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_STPS = K // BLK_K
    PIPE_DEPTH = 2

    WG_NUMBER = 1

    # 寄存器 累计 结果
    acc_per_thread = (BLK_M * BLK_N) // (32 * 4)

    A_layout = mma_shared_layout(
        a_type,
        SwizzleMode.SWIZZLE_128B_ATOM,
        [PIPE_DEPTH, BLK_M, BLK_N],
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

        # smem 就绪
        smem_ready_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)

        # smem 消费完成，可以释放了
        smem_free_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)

        # acc_reg 设置
        acc_ready_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
        acc_free_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)

        Asmem = pool.alloc([PIPE_DEPTH, BLK_M, BLK_K], a_type, layout=A_layout)
        Bsmem = pool.alloc([PIPE_DEPTH, BLK_N, BLK_K], b_type, layout=B_layout)

        pool.commit()

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

        # --- 接下来是 barrier 的初始化 ---
        # 如果在cta 中实际上默认了，cta中的第一个thread才会执行
        if warp_id == 0 and lane_id == 0:
            for s in range(PIPE_DEPTH):
                T.ptx.mbarrier.init(smem_ready_bar.ptr_to([s]), 128)
                T.ptx.mbarrier.init(smem_free_bar.ptr_to([s]), 128)
                T.ptx.mbarrier.init(acc_ready_bar.ptr_to([s]), 128)
                T.ptx.mbarrier.init(acc_free_bar.ptr_to([s]), 128)

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
        elems_per_copy = (
            16 // 2
        )  # 单线程一次 cp.async 搬运的元素数：16B / sizeof(float16)

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
                    B.ptr_to([n_offset + row, k_offset + row]),
                    16,
                )

        """
            MMA 执行过程，
            每个warp负责一个加载计算
        """
        MMA_M = T.meta_var(16)
        MMA_K = T.meta_var(16)
        MMA_N = T.meta_var(8)

        # 以输出的矩阵视角来看
        mma_m_tiles = BLK_M // MMA_M  # 8
        mma_n_tiles = BLK_N // MMA_N  # 16
        k_steps = BLK_K // MMA_K  # 4

        mma_counts = mma_m_tiles * mma_n_tiles  # 总的输出矩阵的块数
        mma_rounds = mma_counts // 4  # 每个warp负责的输出块
        acc_pre_thread_round = (
            MMA_M * MMA_N
        ) // 32  # 4， 每个thread 每一轮负责的累加值

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


        def mma_v2(stage):
            # 换一个迭代的方式, v1的控制流迭代方式
