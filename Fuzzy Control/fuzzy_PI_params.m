FLC_with_rules = readfis('newFLC');
Ti = Kp/Ki
a = Ti
K = Kp/a

output = evalfis(FLC_with_rules,[-1/3 -1/3])

gensurf(FLC_with_rules)