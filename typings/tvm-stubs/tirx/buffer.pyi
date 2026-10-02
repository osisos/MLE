"""TVM 0.26 Buffer typing, including methods attached to ir.Var at runtime."""
from enum import IntEnum
import tvm
from tvm.ir import Expr, PrimType, Type, Var
from tvm.runtime import Object
from . import _buffer_view

class BufferType(Type):
    dtype: PrimType
    storage_scope: str
    shape: list
    strides: list
    elem_offset: tvm.ir.Expr
    data_alignment: int
    offset_factor: int
    layout: object | None
    allocated_addr: list

def is_buffer_var(value) -> bool:
    ...

class BufferAccessKind(IntEnum):
    READ = 1
    WRITE = 2

class Buffer(Var):
    ty: BufferType
    shape: list
    strides: list
    elem_offset: Expr
    data_alignment: int
    offset_factor: int
    layout: object | None
    allocated_addr: list
    dtype: tvm.DataType
    data: Expr

    def access_ptr(self, access_mask, ptr_type='handle', content_lanes=1, offset=0, extent=None):
        ...

    def vload(self, begin, dtype=None, predicate=None):
        ...

    def vstore(self, begin, value, predicate=None):
        ...

    def scope(self):
        ...

    def get_flattened_buffer(self):
        ...

    def with_allocated_addr(self, allocated_addr):
        ...

    def with_dtype(self, dtype):
        ...

    def offset_of(self, indices):
        ...

    @property
    def byte_offset(self):
        ...

    def elem_offset_of(self, indices, inner=True):
        ...

    def byte_offset_of(self, indices, inner=True):
        ...

    def is_scalar(self, alloc_or_decl=True):
        ...

    def ptr_to(self, indices) -> Expr:
        ...

    def view(self, *args, **kwargs) -> 'Buffer':
        ...

    def local(self, *shape, layout=None) -> 'Buffer':
        ...

    def permute(self, *dims) -> 'Buffer':
        ...

    def rearrange(self, pattern: str = ..., /, **sizes) -> 'Buffer':
        ...

    @property
    def sub(self) -> '_buffer_view.SubIndexer':
        ...

    def tile(self, *specs) -> '_buffer_view.TileIndexer':
        ...

    def chunk(self, spec) -> '_buffer_view.ChunkIndexer':
        ...

    def __getitem__(self, indices):
        ...

    # TIR script lowers subscript assignment directly to T.buffer_store.
    def __setitem__(self, indices, value) -> None:
        ...

def decl_buffer(shape, dtype=None, name='buffer', data=None, strides=None, elem_offset=None, scope='', data_alignment=-1, offset_factor=0, span=None, layout='default') -> Buffer:
    ...

def buffer_data(buffer):
    ...

def buffer_data_pointer_type(buffer):
    ...

class DataProducer(Object):
    pass
