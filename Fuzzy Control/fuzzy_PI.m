FLC_with_rules = readfis('FLC');

Ti = Kp/Ki;
a = Ti;
K = Kp/a;

output = evalfis(FLC_with_rules,[0.25 0])

gensurf(FLC_with_rules)