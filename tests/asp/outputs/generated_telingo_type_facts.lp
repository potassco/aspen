type(tel).
expression(tel,(__true,0)).
expression(tel,(__false,0)).
expression(tel,(__or,2)).
expression(tel,(__and,2)).
expression(tel,(__not,1)).
expression(tel,(__initial,0)).
expression(tel,(__final,0)).
expression(tel,(__next,1)).
expression(tel,(__until,2)).
arg((__or,2),0,type,tel).
arg((__or,2),0,safety,unsafe).
arg((__or,2),1,type,tel).
arg((__or,2),1,safety,unsafe).
arg((__and,2),0,type,tel).
arg((__and,2),0,safety,safe).
arg((__and,2),1,type,tel).
arg((__and,2),1,safety,safe).
arg((__not,1),0,type,tel).
arg((__not,1),0,safety,unsafe).
arg((__next,1),0,type,tel).
arg((__next,1),0,safety,safe).
arg((__until,2),0,type,tel).
arg((__until,2),0,safety,unsafe).
arg((__until,2),1,type,tel).
arg((__until,2),1,safety,safe).
var(tel,v1,tel).
var(tel,v2,tel).
subtype(atom,tel).
allow(tel,head).
allow(tel,body).
macro(tel, __wnext(v1), __or(__next(v1), __final)).
macro(tel, __eventually(v1), __until(__true, v1)).
macro(tel, __always(v1), __release(__false, v1)).
macro(tel, __release(v1, v2), __until(v2, __and(v2, __or(v1, __final)))).
