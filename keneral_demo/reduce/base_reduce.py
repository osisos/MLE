from tvm.tirx import script as T

SM_COUNT = 1
WARPS_PER_CTA = 16


def SM89_RRDUCE_SUM(N):
    """规约练习：当前完成线程内与 warp 内求和，尚未合并各 warp 并写入 Y。"""
    elements_per_round = SM_COUNT * WARPS_PER_CTA * 32 * 8
    assert N > 0 and N % elements_per_round == 0, (
        f"暂不处理尾部：N 必须是 {elements_per_round} 的正整数倍"
    )

    @T.prim_func
    def kernel(
        X: T.Buffer((N,), "float16"),
        Y: T.Buffer((1,), "float16"),
    ):
        T.device_entry()

        bx = T.cta_id([SM_COUNT])
        warp_id = T.warp_id([WARPS_PER_CTA])
        lane_id = T.lane_id([32])  # 每个warp中 32个 thread

        # 暂存区reg
        R = T.alloc_local((4,), "float32")
        H = R.view("float16")

        Temp = T.alloc_local((4,), "float32")
        thread_sum = T.alloc_local((1,), "float32")
        thread_sum[0] = T.float32(0)

        @T.inline
        def load_gmem_to_register(src_ptr):
            # 使用 float4指令，一次传入8个f16
            T.ptx.ld(
                src_ptr,
                "uint32",
                "b32",  # b = bit, b32 32位数据，不指定数据内容
                dst=R.ptr_to([0]),
                space="global",
                vec="v4",  # 一次加载多少个上述单元？
            )

        # 实际上如何执行发射。
        # 首先不考虑边界。

        turn = N // elements_per_round
        for r in range(turn):
            # load , stride = 8 * 32 = 256
            idx = (
                ((r * SM_COUNT + bx) * WARPS_PER_CTA + warp_id) * 32 + lane_id
            ) * 8
            load_gmem_to_register(X.ptr_to([idx]))

            # 接下来一个线程内执行规约计算
            for j in T.unroll(4):  # pyright: ignore[reportGeneralTypeIssues]
                Temp[j] = T.Cast("float32", H[2 * j]) + T.Cast("float32", H[2 * j + 1])

            Temp[0] = Temp[0] + Temp[2]
            Temp[1] = Temp[1] + Temp[3]

            Temp[0] = Temp[0] + Temp[1]
            thread_sum[0] = thread_sum[0] + Temp[0]

        # 所有 32 个 lane 一起执行，依次与 lane_id XOR 16、8、4、2、1 交换值。
        # shuffle 只交换寄存器值，后面的加法才执行规约。
        for stage in T.unroll(5):
            thread_sum[0] = thread_sum[0] + T.tvm_warp_shuffle_xor(
                T.uint32(0xFFFFFFFF), thread_sum[0], 16 >> stage, 32, 32
            )
        # 每个 lane 都得到相同的本 warp 总和，供下一步合并各 warp 使用。
        warp_sum = thread_sum[0]

    return kernel
