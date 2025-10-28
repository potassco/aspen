#include "named.lp".

% generate named child at index relation
named_child_idx((S,P),(S,(P,I)),M) :- node((S,P)); named((S,(P,I))); M = #count{ J: named((S,(P,J))), J < I }.
