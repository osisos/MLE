from typing import cast

import tvm
from tvm.ir import Var
from tvm.tirx import script as T
from tvm.tirx.layout import TileLayout, S, laneid, warpid
from tvm.backend.cuda.tile_primitive.tma_utils import mma_shared_layout, SwizzleMode
from tvm.backend.cuda.lang.tile_scheduler import ClusterPersistentScheduler2D

"""
计算矩阵乘法

PIPE_DEPTH = 2

BLK0_M, BLK0_N, BLK0_K = 128, 128, 64

SM89

"""

SM_COUNT = 66

"""
    A (m, k) f16
    B (n, k) f16

    D (m, n) f16

"""


def SM89_GEMM(M, N, K):
    a_type = tvm.DataType("float16")
    b_type = tvm.DataType("float16")
    d_type = tvm.DataType("float32")

    acc_type = tvm.DataType("float32")

    BLK_M, BLK_N, BLK_K = 128, 128, 64
    K_TILES = K // BLK_K
    PIPE_DEPTH = 2

    # TODO: 每个CTA仅使用一个wg
    WG_NUMBER = 1

    # SM89，需要每个线程使用单独的寄存器累加结果
    acc_per_thread = BLK_M * BLK_N // (WG_NUMBER * 4 * 32)  # 每线程保存 128 个累加值

    A_layout = mma_shared_layout(
        a_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_K)
    )
    B_layout = mma_shared_layout(
        b_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_N, BLK_K)
    )
    D_layout = mma_shared_layout(
        d_type, SwizzleMode.SWIZZLE_128B_ATOM, (PIPE_DEPTH, BLK_M, BLK_N)
    )

    # C 的逻辑形状为 [128,128]，对应当前 tile_idx = r * 4 + warp_id 的分工。
    # M 拆为 [8 个 MMA 行块, 2 个寄存器行组, 8 个 lane 行组]；
    # N 拆为 [4 轮列块, 4 个 warp, 4 个 lane 列组, 2 个寄存器列元素]。
    # 逻辑坐标 (m,n) 映射为：
    #   warp = (n // 8) % 4
    #   lane = (m % 8) * 4 + (n % 8) // 2
    #   slot = (m // 16) * 16 + (n // 32) * 4 + ((m // 8) % 2) * 2 + n % 2
    # slot 等价于原来的 r * 4 + i；没有线程轴标记的 stride 表示寄存器槽步长。
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

        # 一维 scope 返回单个 Var；cast 仅收窄 Python 静态类型，不生成 TIR 转换。
        bx = cast(Var, T.cta_id([SM_COUNT]))  # 声明结构，加获取当前线程的cta id
        # bx mean block index
        wg_id = cast(Var, T.warpgroup_id([WG_NUMBER]))  # 1 cta : 1 wg
        warp_id = cast(Var, T.warp_id_in_wg([4]))  # 1: wg : 4 warp
        lane_id = cast(Var, T.lane_id([32]))  # 1 warp : 32 lane

        # 计算当前 warp_group_id 负责的一个输出块

        """
            sm89 仅支持mma.sync，不支持mma.async

            数据流向
                global -> smem -> reg -> mma.sync

            备注：
                mma.sync的含义的指令是，发起调用的时候，warp内的所有线程必须同步发起，这不代表着发起mma后所有线程还在被占用必须同步等待
                如果线程的后续指令和依赖没有关系，完全可以转去执行其他的命令（发射其他的命令）

            必须使用Tirx申请smem 和 reg

        """

        # --- SMEM ---
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

        # TODO: 先不使用Dsmem，尝试直接 reg -> gmem，然后再判断是否需要优化

        pool.commit()

        # --- 接下来是 barrier 的初始化 ---
        # 如果在cta 中实际上默认了，cta中的第一个thread才会执行
        if warp_id == 0 and lane_id == 0:
            for s in range(PIPE_DEPTH):
                T.ptx.mbarrier.init(smem_ready_bar.ptr_to([s]), 128)
                T.ptx.mbarrier.init(smem_free_bar.ptr_to([s]), 128)
                T.ptx.mbarrier.init(acc_ready_bar.ptr_to([s]), 128)
                T.ptx.mbarrier.init(acc_free_bar.ptr_to([s]), 128)

        # Tile scheduler
        tile_scheduler = ClusterPersistentScheduler2D(
            "ts",
            num_m_tiles=M // BLK_M,
            num_n_tiles=N // BLK_N,
            l2_group_size=8,
            num_clusters=SM_COUNT,
        )
        tile_scheduler.init(bx)

        """
            M, N, K = 128, 128, 64
            每个warp group 处理一个矩阵块的乘法

            但是实际上，SM89仅支持[M,N,K] = [16,8,16]的宽度

            因此，这个矩阵的Tile需要切分。

            A = [128, 64] = [8, 4]
            B = [64, 128] = [4, 16]

            因此，实际上每个warp_group_id 负责一个输出分块，因此需要加载一部分A，包含完整的行；一部分B，包含完整的列。

            因为K = 16，按照 K 迭代，类似于外乘的原理

        """

        t_A = T.meta_var((BLK_M * BLK_K * 2) // (4 * 16 * 32))
        t_B = T.meta_var((BLK_N * BLK_K * 2) // (4 * 16 * 32))
        warp_row = T.meta_var((32 * 16) // (BLK_K * 2))  # 每一个warp 搬运的行数
        chunks_per_row = T.meta_var((BLK_K * 2) // 16)
        elements_per_chunk = 16 // 2

        @T.inline
        def load_gmem_to_smem(stage, m_offset, n_offset, k_offset):

            # 每一行给8个thread搬运, 每一轮搬运 (32 * 4)/8 = 16 行，每个warp 搬运4行
            # copy A
            for t in range(t_A):
                # 计算 global row
                row = (
                    t * (BLK_M // t_A)
                    + (warp_id) * warp_row
                    + (lane_id // chunks_per_row)
                )

                # 计算 global col，因为每个row给8个thread搬运
                chunk = lane_id % chunks_per_row

                # 手动推导 128B swizzle（当前 FP16、BLK_K=64）：
                # 每行分成 8 个 16B chunk，每个 chunk 包含 8 个 FP16。
                # smem_chunk = chunk ^ (row % 8)
                # 物理 FP16 下标 = stage * BLK_M * BLK_K + row * BLK_K + smem_chunk * 8
                # XOR 只置换 chunk，片段内的 8 个元素保持连续，保留 16B 对齐。
                # Asmem 已绑定 A_layout，TIRx 会将下面的逻辑坐标转换成上述物理地址；
                # SM89 上也会生成 XOR 地址运算，不依赖 TMA 硬件。
                # 因此 ptr_to 仍传 chunk，不能再传 smem_chunk，否则会重复 swizzle。

                T.ptx.cp_async(
                    Asmem.ptr_to([stage, row, chunk * elements_per_chunk]),
                    A.ptr_to([m_offset + row, k_offset + chunk * elements_per_chunk]),
                    16,
                )

            for t in range(t_B):
                row = t * 16 + warp_id * warp_row + (lane_id // chunks_per_row)

                chunk = lane_id % chunks_per_row  # 8 是怎么来的？8 个 thread 搬运一行

                T.ptx.cp_async(
                    Bsmem.ptr_to([stage, row, chunk * elements_per_chunk]),
                    B.ptr_to([n_offset + row, k_offset + chunk * elements_per_chunk]),
                    16,
                )

        MMA_M = T.meta_var(16)
        MMA_K = T.meta_var(16)
        MMA_N = T.meta_var(8)

        output_tile_rows = T.meta_var(BLK_M // MMA_M)
        output_tile_cols = T.meta_var(BLK_N // MMA_N)

        output_tile_count = T.meta_var(output_tile_rows * output_tile_cols)
        tile_assignment_rounds = T.meta_var(
            output_tile_count // 4
        )  # 每个 warp 处理 32 个小块
        acc_per_mma_per_thread = T.meta_var(
            MMA_M * MMA_N // 32
        )  # 每个小块每线程 4 个累加值

        k_step_count = T.meta_var(BLK_K // MMA_K)

        # 逻辑上是完整的输出 tile，layout 将它分布到 4 个 warp 的寄存器中。
        # 每线程仍只有 128 个 FP32；跨 r、k 和 mma_v1(stage) 调用保留部分和。
        Creg = T.alloc_local((BLK_M, BLK_N), acc_type, layout=C_layout)
        Creg_local = Creg.local()  # 同一份存储的每线程视图，用于清零，不分配新存储。
        # 当前版本的 lowering 不能直接验证带线程轴的 Creg[m,n] 访问，
        # 因此 MMA 使用去掉线程轴的视图，并保留 C_layout 的寄存器排列。
        # 每线程拥有 16 个逻辑行、8 个逻辑列：
        # local_m = 2 * m_tile_idx + i // 2
        # local_n = 2 * (n_tile_idx // 4) + i % 2
        Creg_thread = Creg.local(16, 8, layout=C_layout.storage())

        @T.inline
        def clean_acc_reg():
            for i in range(acc_per_thread):
                Creg_local[i] = T.float32(0.0)

        # 初始化第一个输出 tile。后续接入 scheduler 主循环时，
        # 每个新输出 tile 在完整 K 循环开始前调用一次，不能在切换 stage 时清零。
        clean_acc_reg()

        @T.inline
        def mma_v1(stage):

            for r in range(tile_assignment_rounds):
                tile_idx = r * 4 + warp_id
                # 输出小块坐标在同一个 warp 内一致。
                tile_idx: T.let = r * 4 + warp_id
                m_tile_idx: T.let = tile_idx // output_tile_cols
                n_tile_idx: T.let = tile_idx % output_tile_cols

                # 当前线程视图中的四个结果坐标，按 MMA 的 PTX 顺序排列。
                # ptr_to 通过视图 layout 计算槽地址，不再手写 r * 4 + i。
                c_ptrs = [
                    Creg_thread.ptr_to(
                        [
                            m_tile_idx * 2 + i // 2,
                            (n_tile_idx // 4) * 2 + i % 2,
                        ]
                    )
                    for i in range(acc_per_mma_per_thread)
                ]

                for k in range(k_step_count):
                    # ld A
                    A_smem_m_offset = lane_id % 16
                    A_smem_k_offset = (lane_id // 16) * 8

                    A_reg = T.alloc_local((4,), "uint32")
                    T.ptx.ldmatrix(
                        False,
                        4,
                        ".b16",
                        Asmem.ptr_to(
                            [
                                stage,
                                m_tile_idx * MMA_M + A_smem_m_offset,
                                k * MMA_K + A_smem_k_offset,
                            ]
                        ),
                        *[A_reg.ptr_to([i]) for i in range(4)],
                    )

                    # ld B
                    B_smem_n_offset = lane_id % 8
                    # .x2 使用前 16 个 lane 的源地址，其余 lane 重复有效地址。
                    B_smem_k_offset = ((lane_id // 8) % 2) * 8

                    B_reg = T.alloc_local((2,), "uint32")

                    T.ptx.ldmatrix(
                        False,
                        2,
                        ".b16",
                        Bsmem.ptr_to(
                            [
                                stage,
                                MMA_N * n_tile_idx + B_smem_n_offset,
                                k * MMA_K + B_smem_k_offset,
                            ]
                        ),
                        *[B_reg.ptr_to([i]) for i in range(2)],
                    )

                    # 发起 mma
                    T.ptx.mma(
                        "m16n8k16",
                        "row",
                        "col",
                        "float32",
                        "float16",
                        "float16",
                        "float32",
                        c_ptrs,  # D：当前逻辑小块中，本线程持有的四个结果
                        [A_reg.ptr_to([i]) for i in range(4)],  # A
                        [B_reg.ptr_to([i]) for i in range(2)],  # B
                        c_ptrs,  # C：相同逻辑坐标的部分和，原地累加
                    )

        @T.inline
        def mma_v2(stage):

            for r in range(tile_assignment_rounds):
                tile_idx: T.let = r * 4 + warp_id
                m_tile_idx: T.let = tile_idx // output_tile_cols
                n_tile_idx: T.let = tile_idx % output_tile_cols

                c_ptrs = [Creg_thread.ptr_to([
                    m_tile_idx * 2 +
                ])]

    return kernel
