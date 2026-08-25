from typing import Sequence

from clingo import Control, Function, Number, Symbol, TruthValue, parse_term

clingo_true = TruthValue.True_

K = 1


def expand_ctl(
    control: Control,
    program: str,
    atoms: Sequence[Symbol],
    new_constants: Sequence[Symbol],
) -> None:
    """Add atoms as true externals to ctl, and ground program with expanding index k.

    We require explicit declaration of new constants occurring at the stream position."""
    global K
    clingo_k = Number(K)
    with control.backend() as backend:
        for a_symb in atoms:
            print(f"Adding true external {a_symb}")
            atm_a = backend.add_atom(a_symb)
            backend.add_external(atm_a, value=clingo_true)
            # backend.add_rule([atm_a])
        for new_c in new_constants:
            new_c_symb = Function("new", [new_c, clingo_k])
            print(f"Adding atom {new_c_symb}")
            new_c_atm = backend.add_atom(new_c_symb)
            backend.add_rule([new_c_atm])
    control.ground(parts=[(program, [clingo_k])])
    solve_res = control.solve(on_model=print)
    print(solve_res)
    K += 1


ctl = Control(["0", "--warn", "no-atom-undefined", "--output-debug", "text"])
# ctl = Control(["0", "--warn", "no-atom-undefined"])
ctl.load("../../tests/asp/encodings/expanding/coloring_expand_extended_arith.lp")
# ctl.load("../../tests/asp/encodings/expanding/show.lp")

prg = "color"

ctl.ground()

print("adding vtx(1)...")
print("adding col(2)...")
print("adding vtx(2) arc(1,2)...")
print("adding col(4)...")
expand_ctl(
    ctl,
    prg,
    [
        parse_term("vtx(vtx(1))"),
        parse_term("col(col(1))"),
        parse_term("vtx(vtx(2))"),
        parse_term("arc(vtx(1),vtx(2))"),
        parse_term("col(col(2))"),
    ],
    [
        parse_term("vtx(1)"),
        parse_term("col(1)"),
        parse_term("vtx(2)"),
        parse_term("col(2)"),
    ],
)
print("")
expand_ctl(
    ctl,
    prg,
    [
        parse_term("vtx(vtx(3))"),
        parse_term("arc(vtx(3),vtx(1))"),
        parse_term("col(col(3))"),
    ],
    [parse_term("vtx(3)"), parse_term("col(3)")],
)
print("")
print("releasing arc(vtx(3),vtx(1)) vtx(vtx(3)) col(col(3))...")
ctl.release_external(parse_term("arc(vtx(3),vtx(1))"))
ctl.release_external(parse_term("vtx(vtx(3))"))
ctl.release_external(parse_term("col(col(3))"))
# ctl.assign_external(Function("arc", [Number(1), Number(3)]), False)
# ctl.release_external(Function("col", [Number(2), Number(2)]))
ret = ctl.solve(on_model=print)
print(ret)
print("")
expand_ctl(
    ctl,
    prg,
    [
        parse_term("vtx(vtx(4))"),
        parse_term("arc(vtx(4),vtx(1))"),
        parse_term("col(col(4))"),
    ],
    [parse_term("vtx(4)"), parse_term("col(4)")],
)
print("")
expand_ctl(
    ctl,
    prg,
    [
        parse_term("vtx(vtx(5))"),
        parse_term("arc(vtx(5),vtx(1))"),
        parse_term("col(col(5))"),
    ],
    [parse_term("vtx(5)"), parse_term("col(5)")],
)
