#program metasp_preprocess.

aspen(print(format("#external {}: {}.", (node(Head), (join(", ", Cons), ()))), Source))
  :- ext_head_cons(node(Head), Cons),
		 Cons != (),
		 ext_head_source(Head, Source).

aspen(print(format("#external {}.", (node(Head), ())), Source))
  :- ext_head_cons(node(Head), ()),
		 ext_head_source(Head, Source).
