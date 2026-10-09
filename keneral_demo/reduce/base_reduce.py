from tvm.tirx import script as T

# 本文件使用 TVMScript DSL：@T.prim_func 内的代码会被解析成 IR，
# 并非直接交给 Python 解释器逐行执行。
# T.unroll 返回的 ForFrame 没有 Python __iter__，但 TVM 支持这种 for 语法。
# 因此仅在相关 for 行忽略 Pyright 的 reportGeneralTypeIssues；
# 实际合法性仍由 TVM 解析、lowering 和代码生成检查，不关闭整个文件的类型检查。

SM_COUNT = 66  # 硬件背景信息；当前索引和启动规模都使用 BLOCK_CNT。
WARPS_PER_CTA = 16
BLOCK_CNT = 100  # 启动的 block 数，可以超过 SM 数；TODO：作为配置参数传入。


def SM89_RRDUCE_SUM(N: int):
    """构造 FP16 输入、FP32 累加的规约练习，返回 TIRx PrimFunc。

    一次 host 函数调用会在同一 CUDA stream 上顺序启动两个 kernel：
    第一阶段用 BLOCK_CNT 个 block 写 FP32 局部和；第二阶段只用一个
    block、一个 warp 合并局部和，转换成 FP16 后写入 Y。
    原始输入仍要求完整分块；第二阶段支持 BLOCK_CNT 的尾部。
    """
    # 以下是构造函数中的普通 Python 计算，都会成为编译时常量。
    # 每个线程一轮读 8 个 FP16，一个 block 有 16 * 32 = 512 个线程。
    # 当前配置一轮覆盖 100 * 16 * 32 * 8 = 409600 个输入元素。
    elements_per_round = BLOCK_CNT * WARPS_PER_CTA * 32 * 8

    assert (
        N > 0 and N % elements_per_round == 0
    ), f"暂不处理尾部：N 必须是 {elements_per_round} 的正整数倍"

    @T.prim_func
    def kernel(
        X: T.Buffer((N,), "float16"),
        Y: T.Buffer((1,), "float16"),
    ):
        # 全部 block 共用的 FP32 工作区，每个 block 只写 block_sums[bx]。
        # 分配在两个 device 区域之外，由 host 管理 GPU 全局内存，
        # 生命周期覆盖两次启动，第二阶段可读取第一阶段写出的全部结果。
        block_sums = T.alloc_buffer((BLOCK_CNT,), "float32", scope="global")

        # T.device_entry() 是延伸到函数末尾的平铺标记，不能连写两次来分段。
        # 用显式 attr 作用域表达两个并列 device 区域，SplitHostDevice 会生成
        # 两个 GPU kernel 和顺序调用它们的 host 函数；调用接口仍为 kernel(X, Y)。
        with T.attr(0, "tirx.device_entry", True):
            # 定义执行布局：100 个 block，每 block 16 个 warp，每 warp 32 个 lane。
            # bx 是逻辑 block 编号，并不表示当前运行在哪个 SM 上。
            bx = T.cta_id([BLOCK_CNT])
            warp_id = T.warp_id([WARPS_PER_CTA])
            lane_id = T.lane_id([32])

            # 每个 block 独有一份共享内存，用于在本 block 的 16 个 warp 间传值。
            # pool.commit() 完成共享内存池布局；它不是线程同步屏障。
            pool = T.SMEMPool()
            share_warp_sums = pool.alloc([WARPS_PER_CTA], "float32")
            pool.commit()

            # 每个线程独有 16 字节局部存储，通常映射到寄存器。
            # R 是 4 个 32-bit 槽位；H 是同一存储的 8 个 FP16 元素视图。
            # view 只改变解释方式，不执行数值转换或复制。
            R = T.alloc_local((4,), "float32")
            H = R.view("float16")

            Temp = T.alloc_local((4,), "float32")  # 暂存 8 个输入两两相加后的 4 个和。
            thread_sum = T.alloc_local((1,), "float32")
            thread_sum[0] = T.float32(0)  # 跨所有读取轮次累加，每个线程各有一份。

            warp_sum = T.alloc_local((1,), "float32")
            warp_sum[0] = T.float32(0)

            @T.inline
            def load_gmem_to_register(src_ptr):
                # 用向量加载搬运原始位：4 * 32 bit = 16 字节 = 8 个 FP16。
                # 这里没有把 FP16 转成 uint32 或 FP32 数值；加载后通过 H 解释。
                # 要求源地址 16 字节对齐，且 8 个元素全部有效。
                T.ptx.ld(
                    src_ptr,
                    "uint32",
                    "b32",  # 单个搬运单元为 32 位原始数据。
                    dst=R.ptr_to([0]),
                    space="global",
                    vec="v4",  # 一次加载 4 个 b32 单元。
                )

            turn = N // elements_per_round
            for r in range(turn):
                # 先把 (读取轮次, block, warp, lane) 展平成线程任务编号，再乘 8。
                # 相邻 lane 各负责连续 8 项；相邻读取轮次跨过整个 grid 的数据量。
                # 开头的整除断言保证所有向量读取完整，无需在此做尾部判断。
                idx = (
                    ((r * BLOCK_CNT + bx) * WARPS_PER_CTA + warp_id) * 32 + lane_id
                ) * 8
                load_gmem_to_register(X.ptr_to([idx]))

                # 先把 FP16 数值转换成 FP32，再相加；转换前不做 FP16 加法。
                for j in T.unroll(4):  # pyright: ignore[reportGeneralTypeIssues]
                    Temp[j] = T.Cast("float32", H[2 * j]) + T.Cast(
                        "float32", H[2 * j + 1]
                    )

                Temp[0] = Temp[0] + Temp[2]
                Temp[1] = Temp[1] + Temp[3]

                Temp[0] = Temp[0] + Temp[1]
                thread_sum[0] = thread_sum[0] + Temp[0]

            # 所有 32 个 lane 一起执行，依次与 lane_id XOR 16、8、4、2、1 交换值。
            # shuffle 只交换寄存器值，后面的加法才执行规约。
            for stage in T.unroll(5):  # pyright: ignore[reportGeneralTypeIssues]
                thread_sum[0] = thread_sum[0] + T.tvm_warp_shuffle_xor(
                    T.uint32(0xFFFFFFFF), thread_sum[0], 16 >> stage, 32, 32
                )

            # 每个 lane 都得到相同的本 warp 总和。
            # warp_sum 已初始化为 0，且这里只执行一次，所以加法等价于直接赋值。
            warp_sum[0] = warp_sum[0] + thread_sum[0]

            # 当前 kernel 还未结束。每个 warp 仅派 lane 0 写入共享数组，
            # 避免把同一个 warp 总和重复计入 32 次。
            if lane_id == 0:
                share_warp_sums[warp_id] = warp_sum[0]

            # 所有线程都到达此处后，warp 0 才能读取其他 warp 写出的值。
            # 屏障必须位于 lane_id 条件外；它只同步当前 block，不同步其他 block。
            T.tvm_storage_sync("shared")

            v = T.alloc_local((1,), "float32")
            v[0] = T.float32(0)

            if warp_id == 0:
                # 当前配置：lane 0～15 各读一个 warp 总和，lane 16～31 保持为 0。
                # 边界判断只控制加载；后面的 32-lane 规约仍由整个 warp 执行。
                if lane_id < WARPS_PER_CTA:
                    v[0] = share_warp_sums[lane_id]

                for stage in T.unroll(5):  # pyright: ignore[reportGeneralTypeIssues]
                    v[0] = v[0] + T.tvm_warp_shuffle_xor(
                        T.uint32(0xFFFFFFFF), v[0], 16 >> stage, 32, 32
                    )
                if lane_id == 0:
                    # 一个 block 只写一次自己的槽位，其他 block 写不同的 bx。
                    block_sums[bx] = v[0]

        # 第二次启动：同一 stream 保证第一阶段完成后才执行这里，
        # 无需在两个 launch 之间插入 CPU 等待或 block 内屏障。
        with T.attr(0, "tirx.device_entry", True):
            final_bx = T.cta_id([1])
            final_warp = T.warp_id([1])
            lane_id = T.lane_id([32])

            nums = T.alloc_local((4,), "float32")
            final = T.alloc_local((1,), "float32")
            final[0] = T.float32(0)

            # 每 lane 读 4 项，一个 warp 每轮覆盖 128 项。
            # BLOCK_CNT=100 时 main_turn=0，全部由尾部路径处理。
            main_turn = BLOCK_CNT // 128

            for r in range(main_turn):
                idx = r * 32 * 4 + lane_id * 4
                # 展开 4 次标量读取；不保证生成 float4 向量加载。
                for i in T.unroll(4):  # pyright: ignore[reportGeneralTypeIssues]
                    nums[i] = block_sums[idx + i]

                # 开始规约
                nums[0] = nums[0] + nums[2]
                nums[1] = nums[1] + nums[3]
                nums[0] = nums[0] + nums[1]

                final[0] = final[0] + nums[0]

            # 主流程执行完毕
            # 接下来处理尾部边界

            start = main_turn * 128

            for i in T.unroll(4):  # pyright: ignore[reportGeneralTypeIssues]
                idx = start + lane_id * 4 + i
                if idx < BLOCK_CNT:
                    nums[i] = block_sums[idx]
                else:
                    nums[i] = T.float32(0.0)

            nums[0] = nums[0] + nums[2]
            nums[1] = nums[1] + nums[3]
            nums[0] = nums[0] + nums[1]

            final[0] = final[0] + nums[0]

            # 对跨所有分段的 lane 累计值执行归约，shuffle 输入也必须是 final。
            for stage in T.unroll(5):  # pyright: ignore[reportGeneralTypeIssues]
                final[0] = final[0] + T.tvm_warp_shuffle_xor(
                    T.uint32(0xFFFFFFFF), final[0], 16 >> stage, 32, 32
                )

            if lane_id == 0:
                Y[0] = T.Cast("float16", final[0])

    return kernel
