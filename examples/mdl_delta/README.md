# Parent-pack ΔMDL fixture

Eight observations in two tight clusters.

- The first pack records `delta_mdl_bits = 0`.
- Learning with 4 symbols, then again with 2, keeps the matched codes and quarantines the split codes. The carried symbols keep the description from shrinking, so the child is `reject`.
- Growing from 2 symbols to 4 records a positive delta and `reject`, unless `--mdl-exception` names the reason.
- A tighter re-observation of the same centers, covered in `tests/test_mdl.py`, records a negative delta and `accept`. Matched codes are kept.
