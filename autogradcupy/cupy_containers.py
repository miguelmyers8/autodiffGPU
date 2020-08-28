from __future__ import absolute_import
import cupy as cp
from autograd.extend import _container, primitive
from . import cupy_wrapper as acp
from autograd.core import backward as _backward

_container.__array_priority__ = 102.0


class container(_container):
    __slots__ = []
    __array_priority__ = 1000.0

    @primitive
    def __getitem__(A, idx):
        return A[idx]

    # Constants w.r.t float data just pass though
    shape = property(lambda self: self._value.shape)
    ndim = property(lambda self: self._value.ndim)
    size = property(lambda self: self._value.size)
    dtype = property(lambda self: self._value.dtype)
    T = property(lambda self: acp.transpose(self))

    def __len__(self):
        return len(self._value)

    def astype(self, *args, **kwargs):
        return acp._astype(self, *args, **kwargs)

    def __neg__(self):
        return acp.negative(self)

    def __add__(self, other):
        return acp.add(self, other)

    def __sub__(self, other):
        return acp.subtract(self, other)

    def __mul__(self, other):
        return acp.multiply(self, other)

    def __pow__(self, other):
        return acp.power(self, other)

    def __div__(self, other):
        return acp.divide(self, other)

    def __mod__(self, other):
        return acp.mod(self, other)

    def __truediv__(self, other):
        return acp.true_divide(self, other)

    def __matmul__(self, other):
        return acp.matmul(self, other)

    def __radd__(self, other):
        return acp.add(other, self)

    def __rsub__(self, other):
        return acp.subtract(other, self)

    def __rmul__(self, other):
        return acp.multiply(other, self)

    def __rpow__(self, other):
        return acp.power(other, self)

    def __rdiv__(self, other):
        return acp.divide(other, self)

    def __rmod__(self, other):
        return acp.mod(other, self)

    def __rtruediv__(self, other):
        return acp.true_divide(other, self)

    def __rmatmul__(self, other):
        return acp.matmul(other, self)

    def __eq__(self, other):
        return acp.equal(self, other)

    def __ne__(self, other):
        return acp.not_equal(self, other)

    def __gt__(self, other):
        return acp.greater(self, other)

    def __ge__(self, other):
        return acp.greater_equal(self, other)

    def __lt__(self, other):
        return acp.less(self, other)

    def __le__(self, other):
        return acp.less_equal(self, other)

    def __abs__(self):
        return acp.abs(self)

    def __hash__(self):
        return id(self)
    #===================== CODE SMELL =======================
    def __iadd__(self,other):
        # self += other
        c = anp.add(self , other)
        self._value, self._node = c._value,c._node
        return self

    def __isub__(self,other):
        c = anp.sub(self , other)
        self._value, self._node = c._value,c._node
        return self
    #========================================================
    def backward(self):
        _backward(self)


container.register(cp.ndarray)
types = [
    int,
    float,
    cp.float64,
    cp.float32,
    cp.float16,
    complex,
    cp.complex64,
    cp.complex128,
]
for type_ in types:
    container.register(type_)

# These cupy.ndarray methods are just refs to an equivalent cupy function
nondiff_methods = [
    # 'all',
    # 'any',
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "nonzero",
    # 'searchsorted',
    # 'round'
]
diff_methods = [
    "clip",
    # 'compress',
    "cumprod",
    "cumsum",
    "diagonal",
    # 'max',
    "mean",
    # 'min',
    # 'prod',
    # 'ptp',
    "ravel",
    "repeat",
    "reshape",
    "squeeze",
    "std",
    # 'sum',
    "swapaxes",
    "take",
    "trace",
    "transpose",
    "var",
]
for method_name in nondiff_methods + diff_methods:
    setattr(container, method_name, acp.__dict__[method_name])

# Flatten has no function, only a method.
setattr(container, "flatten", acp.__dict__["ravel"])
