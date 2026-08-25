% grounding from non-expanding, from --output-debug=text with backend atoms
%
% #external vtx(1;3). [true]
% #external col(2;4). [true]
% #external arc(1,3). [true]

{map(3,4)}:-col(4),vtx(3).
{map(1,4)}:-col(4),vtx(1).
{map(3,2)}:-col(2),vtx(3).
{map(1,2)}:-col(2),vtx(1).
:-map(3,4),map(3,2).
:-map(1,4),map(1,2).
:-map(3,2),map(3,4).
:-map(1,2),map(1,4).
mapped(3):-map(3,4).
mapped(1):-map(1,4).
mapped(3):-map(3,2).
mapped(1):-map(1,2).
:-vtx(3),not mapped(3).
:-vtx(1),not mapped(1).
:-arc(1,3),map(3,4),map(1,4).
:-arc(1,3),map(3,2),map(1,2).
