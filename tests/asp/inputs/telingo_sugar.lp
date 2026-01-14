var(v1).
var(v2).



%% defined_as((tel, wnext(v1)), or(next(v1), final))).

match(       sugar("wnext"), (),         ("wnext", 1)).
match(       sugar("wnext"), ((),0),     v1).
replace(     sugar("wnext"), (),         ("or",2)).
replace(     sugar("wnext"), ((),0),     ("next", 1)).
replace(     sugar("wnext"), ((),1),     ("final", 0)).
replace(     sugar("wnext"), (((),0),0), v1).


%% defined_as(tel, eventually(v1), until(true, v1)).

match(       sugar("eventually"), (),      ("eventually", 1)).
match(       sugar("eventually"), ((),0), v1).
replace(     sugar("eventually"), (),     ("until",2)).
replace(     sugar("eventually"), ((),0), ("true", 0)).
replace(     sugar("eventually"), ((),1), v1).

%% defined_as(tel, always(v1), release(false, v1)).

match(       sugar("always"), (),      ("always", 1)).
match(       sugar("always"), ((),0), v1).
replace(     sugar("always"), (),     ("release",2)).
replace(     sugar("always"), ((),0), ("false", 0)).
replace(     sugar("always"), ((),1), v1).

%% defined_as(tel, release(v1,v2), until(v2, and(v2, or(v1, final)))).

match(       sugar("release"), (),              ("release", 2)).
match(       sugar("release"), ((),0),         v1).
match(       sugar("release"), ((),1),         v2).
replace(     sugar("release"), (),             ("until", 2)).
replace(     sugar("release"), ((),0),         v2).
replace(     sugar("release"), ((),1),         ("and", 2)).
replace(     sugar("release"), (((),1),0),     v2).
replace(     sugar("release"), (((),1),1),     ("or", 2)).
replace(     sugar("release"), ((((),1),1),0), v1).
replace(     sugar("release"), ((((),1),1),1), ("final", 0)).
