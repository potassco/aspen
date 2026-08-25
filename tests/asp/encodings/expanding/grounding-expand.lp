% adding vtx(1)...
#external mapped(1,1).
#external mapped(1,2).
mapped(1,1):-mapped(1,2).
mapped(1):-mapped(1,1).
:-vtx(1),not mapped(1).

% adding col(2)...
{map(1,2)}:-col(2),vtx(1).
mapped(1,2):-map(1,2).
#external mapped(1,3).
mapped(1,2):-mapped(1,3).

% adding vtx(3) arc(1,3)...
{map(3,2)}:-col(2),vtx(3).
mapped(3,3):-map(3,2).
#external mapped(3,3).
#external mapped(3,4).
#external mapped(1,4).
mapped(1,3):-mapped(1,4).
mapped(3,3):-mapped(3,4).
mapped(3):-mapped(3,3).
:-vtx(3),not mapped(3).
:-arc(1,3),map(3,2),map(1,2).

% adding col(4)...
{map(1,4)}:-col(4),vtx(1).
{map(3,4)}:-col(4),vtx(3).
mapped(1,4):-map(1,4).
mapped(3,4):-map(3,4).
#external mapped(1,5).
#external mapped(3,5).
mapped(3,4):-mapped(3,5).
mapped(1,4):-mapped(1,5).
:-map(1,4),map(1,2),vtx(1).
:-map(3,4),map(3,2),vtx(3).
:-map(1,2),map(1,4),vtx(1).
:-map(3,2),map(3,4),vtx(3).
:-arc(1,3),map(3,4),map(1,4).
