%% delete(node(Id)) :- aspen(edit(Id,format(FormatStr,Inserts))).

% effects of deleting a node
aspen(delete(node((S,P')))) :- aspen(delete(node((S,P)))), node((S,P')), P'=(I,P).
aspen(delete(type(Id,T))) :- aspen(delete(node(Id))), type(Id,T).
aspen(delete(error(Id))) :- aspen(delete(node(Id))), error(Id).
aspen(delete(missing(Id))) :- aspen(delete(node(Id))), missing(Id).
aspen(delete(extra(Id))) :- aspen(delete(node(Id))), extra(Id).
aspen(delete(leaf_text(Id,Text))) :- aspen(delete(node(Id))), leaf_text(Id,Text).
aspen(delete(field(Id,Name))) :- aspen(delete(node(Id))), field(Id,Name).
