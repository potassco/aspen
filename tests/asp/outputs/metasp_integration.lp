light(l1).
red(L) :- not green(L), light(L).
__next(__eventually(green(L))) :- push(L).
__next(push(l1)) :- __initial.

_show.
_show_term(light(L)) :- green(L).
_show_atom(red(A)) :- red(A).
_show_atom(__next(A)) :- __next(A).
#external push(l1).
#external __initial.
#external green(L): push(L).
