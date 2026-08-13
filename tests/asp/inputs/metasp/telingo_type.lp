#type tel {
    expressions: {
	&true/0;
	&false/0;
	&or/2: {
	    type: tel, safety: unsafe;
	    type: tel, safety: unsafe;
	};
	&and/2: {
	    type: tel, safety: safe;
	    type: tel, safety: safe;
	};
	&not/1: {
			type: tel, safety: unsafe;
	};
	&initial/0;
	&final/0;
	&next/1: {
	    type: tel, safety: safe;
	};
	&until/2: {
	    type: tel, safety: unsafe;
	    type: tel, safety: safe;
	};
    };
    subtypes: {
	atom;
    };
    macros: {
	&wnext(v1): &or(&next(v1), &final);
	&eventually(v1): &until(&true, v1);
	&always(v1): &release(&false, v1);
	&release(v1, v2): &until(v2, &and(v2, &or(v1, &final)));
	where: {
	    v1: tel;
	    v2: tel;
	};
    };
    occurrence: any;
}.
