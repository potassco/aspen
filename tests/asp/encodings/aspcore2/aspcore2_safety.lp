stm(S) :- type(S,("rule";"weak_constraint")).

var(V;V,N) :- type(V,"variable"), leaf_text(V,N).

atoms(S,A) :- child(S,B), type(B,"body"), child(B,NL),
  type(NL,"naf_literal"), child(NL,A), not type(A,"naf").

atoms(S,A) :- child(S,B), type(B,"body"), child(B,A), type(A,"aggregate").

atoms(S,A) :- type(S,"aggregate_element"), child(S,NLS),
  type(NLS,"naf_literals"), child(NLS,NL),
  type(NL,"naf_literal"), child(NL,A), not type(A,"naf").

pos_atoms(S,A) :- atoms(S,A), #false: next_sibling(N,A), type(N,"naf").

eq(A,T,U;A,U,T) :- pos_atoms(_,A), type(A,"builtin_atom"), child(A,U),
   next_sibling(U,R), leaf_text(R,"="), next_sibling(R,T).

eq(A,E,U) :- pos_atoms(_,A), type(A,"aggregate"),
  child(A,B), type(B,"upper"), child(B,R), leaf_text(R,"="),
  child(B,U), next_sibling(R,U), child(A,E), type(E,"aggregate_elements").

desc(S,S) :- atoms(S,_).
desc(arith,B) :- type(B,("binary_operation";"unary_operation")).
desc(T,T;U,U) :- eq(A,T,U).
desc(A,A) :- type(A,"classical_atom").
desc(A,C) :- desc(A,P), child(P,C).

scope(V,"global") :- var(V), stm(S), desc(S,V),
  not desc(S',V): atoms(S',_), desc(S,S'), S'!=S.

scope(V',"global") :- var(V,N), scope(V,"global"),
  stm(S), desc(S,V), desc(S,V'), var(V',N).

scope(V,"local") :- var(V), not scope(V,"global").

binds(S,A,V') :- binds(S,A,V), var(V,N), var(V',N), desc(S,V').

binds(S,A,V) :- pos_atoms(S,A), type(A,"classical_atom"),
  var(V), not desc(arith,V), desc(A,V).

other_binds(S,A,V) :- pos_atoms(S,A),  eq(A,T,_),
  desc(T,V), var(V), binds(S,A',V), A'!=A.

binds(S,A,V) :- pos_atoms(S,A), eq(A,T,U), desc(U,V), var(V),
  scope(V,C), not desc(arith,V),
  other_binds(S,A,V'): var(V'), scope(V',C), V'!=V, desc(T,V').

aspen(exception(node(S),format("Unsafe variables in:\n{}",(node(S),())));
      exception(node(V),format("Variable '{}' is unsafe.",(node(V),())))) :-
  stm(S), desc(S,V), var(V), not binds(_,_,V).
