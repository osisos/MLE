"""TVM 0.26 script exports, including dynamically registered CUDA namespaces."""

from tvm.backend.cuda.script import CUDANamespace, NVSHMEMNamespace, PTXNamespace
from tvm.tirx.lang.alloc_pool import SMEMPool as _SMEMPool, TMEMPool as TMEMPool
from tvm.tirx.buffer import Buffer as _Buffer
from tvm.tirx.layout import Layout as _Layout

from .parser import *
from .parser import Buffer as Buffer, Ptr as Ptr, prim_func as prim_func
from .parser.entry import inline as inline, macro as macro
from . import tile as tile
from .builder.ir import TensorMap as TensorMap, meta_class as meta_class
from .tile import (
    cluster as cluster,
    cta as cta,
    thread as thread,
    warp as warp,
    warpgroup as warpgroup,
    wg as wg,
)

# Registered by tvm.backend.cuda using register_script_namespace at runtime.
ptx: PTXNamespace
cuda: CUDANamespace
nvshmem: NVSHMEMNamespace

class SMEMPool(_SMEMPool):
    # The implementation accepts layout objects as well as the "default" string.
    def alloc(
        self,
        shape,
        dtype="float32",
        strides=None,
        scope="shared.dyn",
        align=0,
        layout: str | _Layout = "default",
    ) -> _Buffer: ...
