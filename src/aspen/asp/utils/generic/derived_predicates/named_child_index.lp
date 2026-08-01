#include "./named.lp".
#include "./child_index.lp".

named_child_index(Child, NamedIdx)
  :- child(Parent,Child), named(Child),
     type(Child, T), not skip_named_child_index(T),
     child_index(Child, Idx),
     NamedIdx = #count{Child': child(Parent, Child'), named(Child'),
     	      		       type(Child',T'), not skip_named_child_index(T'),
     	      		       child_index(Child',Idx'), Idx' < Idx}.
