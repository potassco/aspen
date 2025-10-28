% generate child relation
child((S,P),(S,P')):- node((S,P)); node((S,P')); P'=(P,I).
