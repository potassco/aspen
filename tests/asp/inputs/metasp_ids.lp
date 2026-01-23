#type dummy {
		expressions: {
				&foo/0;
		};
    macros: {
				&foo: &bar;
		};
}.

&p(&q(1,a);&z;#sup) : not -&g.

#show &a/2.
#show &a: &b, c.
