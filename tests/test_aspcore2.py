"""Test for aspcore2 applications of aspen."""

from aspen.utils.testing import AspenTestCase

from .common import aspcore2_lang, encoding_dir

aspcore2_enc_dir = encoding_dir / "aspcore2"


class TestMetaAsp(AspenTestCase):
    """Test suite for metasp-related applications of aspen."""

    maxDiff = None

    def test_aspcore2_list_syntax(self) -> None:
        """Test translation of list function to cons list on nested tuples."""
        self.assert_transform_isomorphic(
            language=aspcore2_lang,
            sources=["p(list(1,2,3))."],
            meta_files=[aspcore2_enc_dir / "aspcore2_list_syntax.lp"],
            expected_sources=["p(cons(1, cons(2, cons(3, nil))))."],
        )

    def test_aspcore2_safety_safe1(self) -> None:
        """Test detection of safety on safe rule."""
        source_str = "p(X,Y) :- q(X), #sum{S,X: r(T,X), S = (2 * T) - X} = Y."
        self.assert_transform_isomorphic(
            language=aspcore2_lang,
            sources=[source_str],
            meta_files=[aspcore2_enc_dir / "aspcore2_safety.lp"],
            expected_sources=[source_str],
        )

    def test_aspcore2_safety_safe2(self) -> None:
        """Test detection of safety on safe rule. The variable S is
        safe, as both X and T are global, and as such the equation
        defining S in the aggregate element is bound."""
        source_str = "p(X,Y) :- q(X), r(T), #sum{S,X: S = (2 * T) - X} = Y."
        self.assert_transform_isomorphic(
            language=aspcore2_lang,
            sources=[source_str],
            meta_files=[aspcore2_enc_dir / "aspcore2_safety.lp"],
            expected_sources=[source_str],
        )

    def test_aspcore2_safety_safe3(self) -> None:
        """Test detection of safety on safe rule. The variable S is
        safe, as both X and T are global, and as such the equation
        defining S in the aggregate element is bound."""
        source_str = "p(X, Y, Z) :- q(X), X = s(Y), f(Z,1) = X + Y."
        self.assert_transform_isomorphic(
            language=aspcore2_lang,
            sources=[source_str],
            meta_files=[aspcore2_enc_dir / "aspcore2_safety.lp"],
            expected_sources=[source_str],
        )

    def test_aspcore2_safety_unsafe1(self) -> None:
        """Test detection of safety on unsafe rule. The rule is unsafe
        as the equation in the aggregate element is not of the correct
        form to bind S."""
        source_str = "p(X,Y) :- q(X), #sum{S,X: r(T,X), S + X = 2 * T} = Y."

        self.assert_transform_raises(
            message_regex=(
                r"s\(0\):1:21-22: Variable 'S' is unsafe.\n"
                r"s\(0\):1:34-35: Variable 'S' is unsafe."
            ),
            language=aspcore2_lang,
            sources=[source_str],
            meta_files=[aspcore2_enc_dir / "aspcore2_safety.lp"],
        )

    def test_aspcore2_safety_unsafe2(self) -> None:
        """Test detection of safety on unsafe rule. The rule is unsafe
        due to default negation of the only atom in the body."""
        source_str = "p(X) :- not r(X)."

        self.assert_transform_raises(
            message_regex=(
                r"s\(0\):1:14-15: Variable 'X' is unsafe.\n"
                r"s\(0\):1:2-3: Variable 'X' is unsafe."
            ),
            language=aspcore2_lang,
            sources=[source_str],
            meta_files=[aspcore2_enc_dir / "aspcore2_safety.lp"],
        )

    def test_aspcore2_safety_unsafe3(self) -> None:
        """Test detection of safety on unsafe rule. The rule is unsafe
        due to default negation of the equality atom."""
        source_str = "p(X,Y) :- q(X), not #sum{ S: r(S) } = Y."

        self.assert_transform_raises(
            message_regex=(
                r"s\(0\):1:38-39: Variable 'Y' is unsafe.\n"
                r"s\(0\):1:4-5: Variable 'Y' is unsafe."
            ),
            language=aspcore2_lang,
            sources=[source_str],
            meta_files=[aspcore2_enc_dir / "aspcore2_safety.lp"],
        )

    def test_aspcore2_choice_sugar(self) -> None:
        """Test translation of choice rules to body aggregate + disjunctive rules."""
        self.assert_transform_isomorphic(
            language=aspcore2_lang,
            sources=["{p(a) : q(2); -p(a) : q(3)} <= 1 :- q(1)."],
            meta_files=[aspcore2_enc_dir / "aspcore2_choice_sugar.lp"],
            expected_sources=[
                (
                    ":- q(1), not #count{ not_choose(p(a)) : p(a), q(2); "
                    "not_choose(-p(a)) : -p(a), q(3) } <= 1.\n"
                    "p(a) | not_choose(p(a)) :- q(1), q(2).\n"
                    "-p(a) | not_choose(-p(a)) :- q(1), q(3)."
                )
            ],
        )

    def test_aspcore2_choice_sugar_reserved(self) -> None:
        """Test that in translation of choice rules to body aggregate
        + disjunctive rules, usage of reserved predicate results in
        exception."""
        self.assert_transform_raises(
            message_regex=(r"s\(0\):1:42-55: Predicate not_choose/1 is system reserved."),
            language=aspcore2_lang,
            sources=["{p(a) : q(2); -p(a) : q(3)} <= 1 :- q(1). not_choose(a)."],
            meta_files=[aspcore2_enc_dir / "aspcore2_choice_sugar.lp"],
        )

    def test_aspcore2_choice_lower_bound_sugar(self) -> None:
        """Test translation of choice rules lower bound to upper bound"""
        self.assert_transform_isomorphic(
            language=aspcore2_lang,
            sources=["1 <= {a;b}."],
            meta_files=[aspcore2_enc_dir / "aspcore2_choice_lower_sugar.lp"],
            initial_program=("aspcore_choice_rewrite_lower", ()),
            expected_sources=[" {a;b} >= 1."],
        )

    def test_aspcore2_choice_double_bound_sugar(self) -> None:
        """Test translation of choice rules lower and upper bound"""
        self.assert_transform_isomorphic(
            language=aspcore2_lang,
            sources=["1 <= {a;b} < 2 :- c."],
            meta_files=[aspcore2_enc_dir / "aspcore2_choice_lower_sugar.lp"],
            initial_program=("aspcore_choice_double_bound", ()),
            expected_sources=["1 <= { a;b } :- c.\n{ a;b } < 2 :- c."],
        )

    def test_aspcore2_choice_bound_sugar(self) -> None:
        """Test translation of choice rule bounds to only choice rules
        with upper bound."""
        self.assert_transform_isomorphic(
            language=aspcore2_lang,
            sources=["1 <= {a;b} < 2 :- c."],
            meta_files=[aspcore2_enc_dir / "aspcore2_choice_lower_sugar.lp"],
            meta_string=(
                "#program aspcore_choice_double_bound."
                'aspen(next_program("aspcore_choice_rewrite_lower",())).'
            ),
            initial_program=("aspcore_choice_double_bound", ()),
            expected_sources=[" { a;b } >= 1 :- c.\n{ a;b } < 2 :- c."],
        )
