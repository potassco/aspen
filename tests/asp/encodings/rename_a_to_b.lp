#program rename_x_to_y(x,y).

% why doesn't this rule fire with leaf(N,x) in the body?
aspen(edit(node(N),y)) :- node(N), type(N,"identifier"), leaf_text(N,"a").

aspen(next_program("add_var_to_atoms", ("X",))).
