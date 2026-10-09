"""检查两阶段归约；加 --run-cuda 在 GPU 上验证数值。

仓库根目录运行：
    .venv/bin/python -m keneral_demo.reduce.verify_base_reduce
    .venv/bin/python -m keneral_demo.reduce.verify_base_reduce --run-cuda
"""

import argparse

import numpy as np
import tvm
from tvm.ir import Call
from tvm.tirx.expr import StringImm

from . import base_reduce


def check_launches(func, block_count):
    # 必须指定 host target，pipeline 才会真正拆出 host 调用和 device kernel。
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_89"}, host="llvm")
    mod = tvm.tirx.transform.BindTarget(target)(tvm.IRModule({"kernel": func}))
    pipeline, _, _ = tvm.tirx.get_tir_pipeline("tirx")
    mod = pipeline(mod)
    devices = {
        gv.name_hint: f
        for gv, f in mod.functions.items()
        if f.attrs.get("calling_conv") == 2
    }
    assert len(devices) == 2, "应生成两个独立 GPU kernel"
    launches = []

    def visit(node):
        if (
            isinstance(node, Call)
            and isinstance(node.op, tvm.ir.Op)
            and node.op.name == "tirx.tvm_call_packed"
            and isinstance(node.args[0], StringImm)
            and node.args[0].value in devices
        ):
            launches.append(node)

    tvm.tirx.stmt_functor.post_order_visit(mod["kernel"].body, visit)
    assert len(launches) == 2
    first, second = launches
    assert first.args[0].value != second.args[0].value
    assert [int(x) for x in first.args[3:5]] == [block_count, 512]
    assert [int(x) for x in second.args[3:5]] == [1, 32]
    # Buffer.data 在此版本中是表达式；比较其引用的 buffer，而非表达式对象身份。
    tvm.ir.assert_structural_equal(first.args[2], second.args[2])

    module = tvm.tirx.build(func, target=target, pipeline="tirx")
    source = module.imports[0].inspect_source()
    assert "__launch_bounds__(512)" in source
    assert "__launch_bounds__(32)" in source
    return module


def check_cuda(module, n):
    device = tvm.cuda(0)
    x = tvm.runtime.empty((n,), "float16", device=device)
    y = tvm.runtime.empty((1,), "float16", device=device)
    # 二进制精确可表示的输入，结果均在 FP16 有限范围内。
    # 连续复用输入/输出，覆盖非零数据、仅最后一个 block 有数据、全零数据。
    constant = np.full(n, 1 / 64, dtype=np.float16)
    signed = ((np.arange(n) % 17 - 8) / 64).astype(np.float16)
    sparse = np.zeros(n, dtype=np.float16)
    sparse[-4096:] = 1 / 64
    for values in (constant, signed, sparse, np.zeros_like(sparse)):
        x.copyfrom(values)
        y.copyfrom(np.array([np.nan], dtype=np.float16))
        module["kernel"](x, y)
        device.sync()
        expected = np.array([values.sum(dtype=np.float64)], dtype=np.float16)
        np.testing.assert_array_equal(y.numpy(), expected)


def main(run_cuda=False):
    if run_cuda and (not tvm.runtime.enabled("cuda") or not tvm.cuda(0).exist):
        raise RuntimeError("--run-cuda 需要启用 CUDA 的 TVM 和 NVIDIA GPU")

    original_count = base_reduce.BLOCK_CNT
    try:
        # 仅尾段、整主段、主段加尾段、多个主段；并覆盖一轮和两轮输入。
        for count in (100, 128, 129, 256):
            base_reduce.BLOCK_CNT = count
            for rounds in (1, 2):
                n = count * base_reduce.WARPS_PER_CTA * 32 * 8 * rounds
                module = check_launches(base_reduce.SM89_RRDUCE_SUM(n), count)
                if run_cuda:
                    check_cuda(module, n)
                print(f"PASS: BLOCK_CNT={count}, rounds={rounds}")
    finally:
        base_reduce.BLOCK_CNT = original_count

    print("PASS: 两次启动的顺序、规模、共享工作区和 CUDA 源码生成")
    print("PASS: GPU 数值验证" if run_cuda else "未运行 GPU 数值验证；可添加 --run-cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-cuda", action="store_true")
    main(parser.parse_args().run_cuda)
