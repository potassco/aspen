#include "../derived_predicates/child_index.lp".

aspen(subtree(Query, Root, (), Root; Query, Root', (), Root'))
  :- aspen(query(Query)),
     Query=isomorphic(Root, Root').

aspen(subtree(Query, Root, (Path, Idx), Child))
  :- node(Node),
		 aspen(subtree(Query, Root, Path, Node)),
		 child(Node, Child),
     child_index(Child, Idx).

aspen(iso_query_member(Query, 0, Root; Query, 1, Root'))
  :- aspen(query(Query)),
     Query=isomorphic(Root, Root').

aspen(return(Query, missing_path(Root', Path)))
  :- aspen(iso_query_member(Query, Idx, Root)),
		 aspen(iso_query_member(Query, Idx', Root')),
		 Idx != Idx',
		 aspen(subtree(Query, Root,  Path, Node)),
     not aspen(subtree(Query, Root', Path, _)).

#defined extra/1.
#defined missing/1.
#defined error/1.
#defined field/2.

attr(Node, type(T)) :- type(Node, T).
attr(Node, error) :- error(Node).
attr(Node, missing) :- missing(Node).
attr(Node, extra) :- extra(Node).
attr(Node, leaf_text(T)) :- leaf_text(Node, T).
attr(Node, field(Field)) :- field(Node, Field).
attr(Node, source_root(Source)) :- source_root(Source, Node).

% every path should have exactly one node identifier.
aspen(return(Query, duplicate_nodes_at_path(Path, Node)))
  :- aspen(iso_query_member(Query, Idx, Root)),
		 aspen(subtree(Query, Root, Path, Node)),
		 aspen(subtree(Query, Root, Path, Node')),
		 Node != Node'.

% there should not be any dangling tree facts, that is, facts that
% contain a term that is not a node id which occurs at some path in
% the subtree
aspen(return(Query, dangling_node(Node)))
  :- node(Node),
		 aspen(iso_query_member(Query, _, _)),
		 not aspen(subtree(Query, _, _, Node)).

aspen(return(Query, dangling_attr(Node, Attr)))
  :- attr(Node, Attr),
		 aspen(iso_query_member(Query, _, _)),
		 not aspen(subtree(Query, _, _, Node)).

aspen(return(Query, dangling_child(Parent, Child)))
  :- child(Parent, Child),
		 aspen(iso_query_member(Query, _, _)),
		 #false: aspen(subtree(Query, _, _, Parent)),
             aspen(subtree(Query, _, _, Child)).

aspen(return(Query, dangling_next(Node, NextNode)))
  :- next_sibling(Node, NextNode),
		 aspen(iso_query_member(Query, _, _)),
		 #false: aspen(subtree(Query, _, _, Node)),
             aspen(subtree(Query, _, _, NextNode)).

% if a node id in one tree has an attribute, then the corresponding
% node id in the other tree at the same path should also have the
% attribute
aspen(return(Query, missing_attr(Node', Attr)))
  :- aspen(iso_query_member(Query, Idx, Root)),
		 aspen(iso_query_member(Query, Idx', Root')),
		 Idx != Idx',
		 aspen(subtree(Query, Root, Path, Node)),
		 aspen(subtree(Query, Root', Path, Node')),
		 attr(Node, Attr), not attr(Node', Attr).

% if two node ids are connected via the
% next_sibling/next_named_sibling relation, then the corresponding
% node ids in the other tree must also be connected via the same
% relation
aspen(return(Query, missing_next_sibling(Path, NextPath)))
  :- aspen(iso_query_member(Query, Idx, Root)),
		 aspen(iso_query_member(Query, Idx', Root')),
		 Idx != Idx',
		 aspen(subtree(Query, Root, Path, Node)),
		 next_sibling(Node, NextNode),
		 aspen(subtree(Query, Root, NextPath, NextNode)),
		 #false: aspen(subtree(Query, Root', Path, Node')),
		         aspen(subtree(Query, Root', NextPath, NextNode')),
						 next_sibling(Node', NextNode').

aspen(return(Query, isomorphic))
  :- aspen(query(Query)),
		 Query=isomorphic(Root, Root'),
		 not aspen(return(Query, missing_attr(_,_))),
		 not aspen(return(Query, missing_path(_, _))),
		 not aspen(return(Query, missing_next_sibling(_,_))),
     not aspen(return(Query, dangling_node(_))),
     not aspen(return(Query, dangling_child(_,_))),
     not aspen(return(Query, dangling_next(_,_))),
     not aspen(return(Query, dangling_attr(_,_))).
