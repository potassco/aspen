light(l1).
red(L) :- not green(L), light(L).
&next(&eventually(green(L))) :- push(L).
&next(push(l1)) :- &initial.

#show.
#show light(L): green(L).
#show red/1.
#show &next/1.
