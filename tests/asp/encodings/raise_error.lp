aspen(exception(node(N),
								format("This is an error for node '{}'.",
											 (node(N), ()))))
  :- leaf_text(N,"a").
