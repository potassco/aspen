"""
The aspen project.
"""

from typing import Sequence, Tuple

from clingo import Control, Function, Number, Symbol, TruthValue, parse_term

clingo_true = TruthValue.True_

K = 1


def expand_ctl(
    control: Control,
    program: str,
    atoms: Sequence[Tuple[str, list[Symbol]]],
) -> None:
    """Add atoms as true externals to ctl, and ground program with expanding index k."""
    global K
    clingo_k = Number(K)
    with control.backend() as backend:
        for name, args in atoms:
            a_symb = Function(name, args)
            # print(f"Adding true external {a_symb}")
            atm_a = backend.add_atom(a_symb)
            backend.add_external(atm_a, value=clingo_true)
            # backend.add_rule([atm_a])

    control.ground(parts=[(program, [clingo_k])])
    print("ANSWER SETS...")
    solve_res = control.solve(on_model=print)
    print("END OF ANSWER SETS")
    print(solve_res)
    with control.backend() as backend:
        for s in ctl.symbolic_atoms.by_signature("__new", 1):
            symb = s.symbol
            # print(f"found new symbol {symb}")
            atm = backend.add_atom(Function("__defined", [symb.arguments[0]]))
            backend.add_rule([atm])
            ctl.release_external(symb)
    K += 1


# ctl = Control(["0", "--warn", "no-atom-undefined", "--output-debug", "text"])
ctl = Control(["0", "--warn", "no-atom-undefined"])
ctl.load("../../tests/asp/encodings/expanding/coloring_initial.lp")
ctl.load("../../tests/asp/encodings/expanding/coloring_expand_k_new.lp")
prg = "color"

# expand_ctl(ctl, prg, [("vtx", [Number(1)])])

# expand_ctl(ctl, prg, [("col", [Number(1)])])

# expand_ctl(ctl, prg, [("vtx", [Number(2)])])

# expand_ctl(ctl, prg, [("arc", [Number(1), Number(2)])])

# expand_ctl(ctl, prg, [("col", [Number(2)])])

expand_ctl(
    ctl,
    prg,
    [("vtx", [Number(1)])]
    + [("col", [Number(1)])]
    + [("vtx", [Number(2)])]
    + [("arc", [Number(1), Number(2)])]
    + [("col", [Number(2)])],
)
print("releasing vtx(1) arc(1,2).")
ctl.release_external(Function("arc", [Number(1), Number(2)]))
ctl.release_external(Function("vtx", [Number(1)]))
ret = ctl.solve(on_model=print)
print(ret)

expand_ctl(ctl, prg, [("vtx", [Number(3)]), ("arc", [Number(3), Number(2)])])

expand_ctl(ctl, prg, [("vtx", [Number(4)]), ("arc", [Number(4), Number(2)])])
