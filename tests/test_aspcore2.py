from aspen.utils.testing import AspenTestCase

from .common import aspcore2_lang, encoding_dir, input_dir, output_dir


class TestMetaAsp(AspenTestCase):
    """Test suite for metasp-related applications of aspen."""

    maxDiff = None

    def test_aspcore2_list_syntax(self) -> None:
        """Test  removal of & from metasp"""
        self.assert_transform_isomorphic(
            language=aspcore2_lang,
            sources=["p(list(1,2,3))."],
            meta_files=[encoding_dir / "aspcore2_list_syntax.lp"],
            expected_sources=["p(cons(1, cons(2, cons(3, nil))))."],
        )
