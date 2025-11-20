aspen(log(node(N),"warning",format("This is a log for node '{0}'.", (node(N),()))))
  :- type(N,"rule").
aspen(log("warning", "This is a log without location.")).
