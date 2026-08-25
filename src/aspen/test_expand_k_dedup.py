"""
The aspen project.
"""

from typing import Sequence, Tuple

from clingo import Control, Function, Number, Symbol, TruthValue, parse_term

clingo_true = TruthValue.True_

K = 1

SIGS = [
    ("col", 1),
    ("vtx", 1),
    ("map", 2),
    ("mapped", 1),
    ("has_in_edge", 1),
    ("index", 2),
]


def expand_ctl(
    control: Control,
    program: str,
    atoms: Sequence[Tuple[str, list[Symbol]]],
    initial: bool = False,
) -> None:
    """Add atoms as true externals to ctl, and ground program with expanding index k."""
    global K
    global sig2defined
    clingo_k = Number(K)
    with control.backend() as backend:
        for name, args in atoms:
            a_symb = Function(name, args)
            # print(f"Adding true external {a_symb}")
            atm_a = backend.add_atom(a_symb)
            backend.add_external(atm_a, value=clingo_true)

    control.ground(parts=[(program, [clingo_k])])
    print("ANSWER SETS...")
    solve_res = control.solve(on_model=print)
    print("END OF ANSWER SETS")
    print(solve_res)
    with control.backend() as backend:
        for sig in SIGS:
            defined_atoms = sig2defined[sig]
            symbs = {
                s.symbol
                for s in ctl.symbolic_atoms.by_signature(*sig)
                if not s.is_external
            }
            for symb in symbs.difference(defined_atoms):
                defined_symb = Function("defined", [symb])
                # print(f"adding fact {defined_symb}")
                atm = backend.add_atom(defined_symb)
                backend.add_rule([atm])
            defined_atoms.update(symbs)
    K += 1


# ctl = Control(["0", "--warn", "no-atom-undefined", "--output-debug", "text"])
ctl = Control(["0", "--warn", "no-atom-undefined"])
ctl.load("../../tests/asp/encodings/expanding/coloring_initial.lp")
ctl.load("../../tests/asp/encodings/expanding/coloring_expand_k_dedup.lp")
prg = "color"
sig2defined: dict[tuple[str, int], set[Symbol]] = {sig: set() for sig in SIGS}

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

# Why does the external release/assignment still work if the final k argument is not given?
# It might be that they are assigned the same solver literal.
print("releasing vtx(1) arc(1,2).")
ctl.release_external(Function("arc", [Number(1), Number(2)]))
ctl.release_external(Function("vtx", [Number(1)]))
ret = ctl.solve(on_model=print)
print(ret)

expand_ctl(ctl, prg, [("vtx", [Number(3)]), ("arc", [Number(3), Number(2)])])

expand_ctl(ctl, prg, [("vtx", [Number(4)]), ("arc", [Number(4), Number(2)])])
