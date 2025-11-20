% built-in types: atom, number, string, constant, function

type(tel).
allow(tel, head).
allow(tel, body).

formula((tel, atom)).
formula((tel, (true, 0))).
formula((tel, (false, 0)).
formula((tel, (not, 1))).
  arg((tel, (not, 1)), 0, type, tel).
formula((tel, (and, 2))).
	arg((tel, (and, 2)), 0, type, tel).
	arg((tel, (and, 2)), 1, type, tel).
	arg((tel, (and, 2)), 0, safety, safe).
	arg((tel, (and, 2)), 1, safety, safe).
formula((tel, (or, 2))).
	arg((tel, (or, 2)), 0, type, tel).
	arg((tel, (or, 2)), 1, type, tel).

formula((tel, (final, 0))).
formula((tel, (next, 1))).
	arg((tel, (next, 1)), 0, type, tel).
	arg((tel, (next, 1)), 0, safety, safe).
formula((tel, (until, 2)).
	arg((tel, (until, 2)), 0, type, tel).
	arg((tel, (until, 2)), 0, safety, unsafe).
	arg((tel, (until, 2)), 1, type, tel).
	arg((tel, (until, 2)), 1, safety, safe).

formula((tel, (initial, 0))).
formula((tel, (previous, 1))).
	arg((tel, (previous, 1)), 0, type, tel).
	arg((tel, (previous, 1)), 0, safety, safe).
formula((tel, (since, 2)).
	arg((tel, (since, 2)), 0, type, tel).
	arg((tel, (since, 2)), 0, safety, unsafe).
	arg((tel, (since, 2)), 1, type, tel).
	arg((tel, (since, 2)), 1, safety, safe).
