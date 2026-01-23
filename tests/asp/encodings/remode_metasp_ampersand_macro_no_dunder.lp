no_dunder(Id)
  :- ampersand_id(_, Id),
		 child(Node, Id),
		 type(Node, "metasp_symbolic_atom"),
		 leaf_text(Id, "macro").
