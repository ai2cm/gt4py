import dace
import hypothesis.strategies as hyp_st
import numpy as np
import pytest

from gt4py import gtscript
from gt4py.gtscript import PARALLEL, computation, interval


@pytest.fixture
def dace_stencil():
    @gtscript.stencil(backend="gtc:dace")
    def defn(inp: gtscript.Field[np.float64], outp: gtscript.Field[np.float64]):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa F841: local variable 'outp' is assigned to but never used

    return defn


def tuple_st(min_value, max_value):
    return hyp_st.tuples(
        hyp_st.integers(min_value=min_value, max_value=max_value),
        hyp_st.integers(min_value=min_value, max_value=max_value),
        hyp_st.integers(min_value=min_value, max_value=max_value),
    )


@pytest.mark.parametrize("domain", [(0, 2, 3), (3, 3, 3), (1, 1, 1)])
@pytest.mark.parametrize("outp_origin", [(0, 0, 0), (7, 7, 7), (2, 2, 0)])
def test_origin_offsetting(dace_stencil, domain, outp_origin):

    frozen_stencil = dace_stencil.freeze(
        domain=domain, origin={"inp": (0, 0, 0), "outp": outp_origin}
    )

    inp = np.full(fill_value=7.0, dtype=np.float64, shape=(10, 10, 10))
    outp = np.zeros(dtype=np.float64, shape=(10, 10, 10))

    @dace.program
    def call_frozen_stencil():
        frozen_stencil(inp=inp, outp=outp)

    call_frozen_stencil()

    assert np.allclose(inp, 7.0)
    assert np.allclose(
        outp[
            outp_origin[0] : outp_origin[0] + domain[0],
            outp_origin[1] : outp_origin[1] + domain[1],
            outp_origin[2] : outp_origin[2] + domain[2],
        ],
        7.0,
    )
    assert np.sum(outp, axis=(0, 1, 2)) == np.prod(domain) * 7.0


def test_optional_arg_noprovide():
    @gtscript.stencil(backend="gtc:dace")
    def stencil(
        inp: gtscript.Field[np.float64],
        outp: gtscript.Field[np.float64],
        unused_field: gtscript.Field[np.float64],
        unused_par: float,
    ):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa F841: local variable 'outp' is assigned to but never used

    frozen_stencil = stencil.freeze(
        domain=(3, 3, 10), origin={"inp": (2, 2, 0), "outp": (2, 2, 0), "unused_field": (0, 0, 0)}
    )

    inp = np.full(fill_value=7.0, dtype=np.float64, shape=(10, 10, 10))
    outp = np.zeros(dtype=np.float64, shape=(10, 10, 10))

    @dace.program
    def call_frozen_stencil():
        frozen_stencil(inp=inp, outp=outp)

    call_frozen_stencil()

    assert np.allclose(inp, 7.0)
    assert np.allclose(
        outp[
            2:5,
            2:5,
            :,
        ],
        7.0,
    )
    assert np.sum(outp, axis=(0, 1, 2)) == 90 * 7.0


# @pytest.mark.xfail(reason="Optional arguments can't be keyword arguments in dace parsing.")
def test_optional_arg_provide():
    @gtscript.stencil(backend="gtc:dace")
    def stencil(
        inp: gtscript.Field[np.float64],
        unused_field: gtscript.Field[np.float64],
        outp: gtscript.Field[np.float64],
        unused_par: float,
    ):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa F841: local variable 'outp' is assigned to but never used

    frozen_stencil = stencil.freeze(
        domain=(3, 3, 10), origin={"inp": (2, 2, 0), "outp": (2, 2, 0), "unused_field": (0, 0, 0)}
    )

    inp = np.full(fill_value=7.0, dtype=np.float64, shape=(10, 10, 10))
    outp = np.zeros(dtype=np.float64, shape=(10, 10, 10))
    unused_field = np.zeros(dtype=np.float64, shape=(10, 10, 10))

    @dace.program
    def call_frozen_stencil():
        frozen_stencil(inp, unused_field, outp, 7.0)

    call_frozen_stencil()

    assert np.allclose(inp, 7.0)
    assert np.allclose(
        outp[
            2:5,
            2:5,
            :,
        ],
        7.0,
    )
    assert np.sum(outp, axis=(0, 1, 2)) == 90 * 7.0


@pytest.mark.xfail(
    reason="DaCe catches all exceptions in __sdfg__ interface, hence the Exception Type and string don't"
    " match."
)
def test_nondace_raises():
    @gtscript.stencil(backend="gtc:numpy")
    def numpy_stencil(inp: gtscript.Field[np.float64], outp: gtscript.Field[np.float64]):
        with computation(PARALLEL), interval(...):
            outp = inp  # noqa F841: local variable 'outp' is assigned to but never used

    frozen_stencil = numpy_stencil.freeze(
        domain=(3, 3, 3), origin={"inp": (0, 0, 0), "outp": (0, 0, 0)}
    )

    inp = np.full(fill_value=7.0, dtype=np.float64, shape=(10, 10, 10))
    outp = np.zeros(dtype=np.float64, shape=(10, 10, 10))

    @dace.program
    def call_frozen_stencil():
        frozen_stencil(inp=inp, outp=outp)

    with pytest.raises(
        TypeError,
        match="Only dace backends are supported in DaCe-orchestrated programs."
        ' (found "gtc:numpy")',
    ):
        call_frozen_stencil()
