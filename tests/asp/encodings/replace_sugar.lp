% This encoding replaces syntactic sugar

% Matching syntactic sugar

found_match(SugarId, (), NodeId, NodeId)
  :- match(SugarId, (), (Name, Arity)),
		 symbol_signature(NodeId, Name, Arity).

possible_match(SugarId, Path', RootId, NodeChildId)
  :- found_match(SugarId, Path, RootId, NodeId),
		 match(SugarId, Path', Pattern),
		 Path' = (Path, Idx),
		 child(NodeId, TermsId),
		 type(TermsId, "terms"),
		 named_child_idx(TermsId,NodeChildId, Idx).

found_match(SugarId, Path, RootId, NodeId)
  :- possible_match(SugarId, Path, RootId, NodeId),
		 match(SugarId, Path, (Name, Arity)),
		 symbol_signature(NodeId, Name, Arity).

bound_var(SugarId, Path, RootId, (Var, NodeId))
  :- possible_match(SugarId, Path, RootId, NodeId),
		 match(SugarId, Path, Var),
		 var(Var).

% variable always matches
found_match(SugarId, Path, RootId, NodeId)
  :- bound_var(SugarId, Path, RootId,  (Var, NodeId)).

% Replacing syntactic sugar

replace_child(SugarId, Path, Path')
  :- replace(SugarId, Path, _),
		 replace(SugarId, Path', _),
		 Path' = (Path, Idx).

replace_leaf(SugarId, Path)
  :- replace(SugarId, Path, _),
		 not replace_child(SugarId, Path, _).

% Base case - replace leaf
replace_text(SugarId, RootId, Path, NodeId')
  :- found_match(SugarId, (), RootId, RootId),
		 replace_leaf(SugarId, Path),
		 replace(SugarId, Path, Var),
		 var(Var),
		 bound_var(SugarId, _, RootId, (Var, NodeId')).

replace_text(SugarId, RootId, Path, Name)
  :- found_match(SugarId, (), RootId, RootId),
		 replace_leaf(SugarId, Path),
		 replace(SugarId, Path, Expression),
		 Expression = (Name, Arity),
     Arity = 0.

% Recursive case

replace_child_last(SugarId, Path, (Path, Idx))
  :- replace(SugarId,Path,_),
		 replace_child(SugarId, Path, (Path, Idx)),
		 not replace_child(SugarId, Path, (Path, Idx+1)).

replace_child_text_cons_acc(SugarId, RootId, Path, Idx, (ChildText, ()))
  :- replace_child_last(SugarId, Path, Path'),
		 Path' = (Path, Idx),
     replace_text(SugarId, RootId, Path',ChildText).

replace_child_text_cons_acc(SugarId, RootId, Path, Idx - 1, (ChildText, Cons))
  :- replace_child_text_cons_acc(SugarId, RootId, Path, Idx, Cons),
		 Path' = (Path, Idx - 1),
		 replace_child(SugarId, Path, Path'),
		 replace_text(SugarId, RootId, Path', ChildText).

replace_child_text_cons(SugarId, RootId, Path, Cons)
  :- replace_child_text_cons_acc(SugarId, RootId, Path, 0, Cons).

replace_text(SugarId, RootId, Path, format("{}({})", (Name, (join(", ", Cons), ()))))
  :- replace_child_text_cons(SugarId, RootId, Path, Cons),
		 replace(SugarId, Path, (Name, _)).

replace_text_root(SugarId, RootId, Text)
  :- replace_text(SugarId, RootId, (), Text).

aspen(edit(RootId, Text))
  :- replace_text_root(SugarId, RootId, Text).

% repeat transform again if there was a match, in case syntactic sugar
% generates new syntactic sugar.
%% aspen(next_program("base", (,))) :- found_match(_,_,_,_).

%% #show found_match/4.
%% #show possible_match/4.
%% #show bound_var/4.
%% #show replace_leaf/2.
%% #show replace_text/4.
%% #show replace_child/3.
%% #show replace_child_last/3.
%% #show replace_child_text_cons/4.
%% #show replace_text_root/3.
%% #show aspen/1.
