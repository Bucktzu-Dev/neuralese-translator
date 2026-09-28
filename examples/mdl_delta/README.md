# Parent-pack ΔMDL fixture

Eight observations in two tight clusters. Learning with 4 symbols, then again with 2, records a negative `delta_mdl_bits` (`child.mdl_bits - parent.mdl_bits`). The reverse step records a positive delta and `decision=reject` unless `--mdl-exception` names the reason.

The first pack, which has no parent, records `delta_mdl_bits = 0`.
