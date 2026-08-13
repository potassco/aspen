#program rename_x_to_y(x,y).

aspen(replace(node(N),y)) :- node(N), type(N,"identifier"), leaf_text(N,"a").

aspen(next_program("add_var_to_atoms", ("X",))).
