import dace
from .test_gtc.oir_utils import StencilFactory
from gtc.dace.oir_to_dace import OirSDFGBuilder
from .test_gtc.test_dace import assert_sdfg_equal
def test_serialize_dace_oir():
    oir = StencilFactory()
    orig_sdfg = OirSDFGBuilder().visit(oir)

    loaded_sdfg = dace.SDFG.from_json(orig_sdfg.to_json())
    assert_sdfg_equal(orig_sdfg, loaded_sdfg)