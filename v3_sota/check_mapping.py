import sys, json, h5py, numpy as np
clip_path, ref_path = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else None
with h5py.File(clip_path) as c, h5py.File(ref_path) as r:
    cn = {k: int(c[k]['picks'][-1]) + 1 for k in c}
    rn = {k: int(r[k]['n_frames'][()]) for k in r}
bad, mapping = 0, {}
print(f'{"clip key":<12}{"clip frames":>12}{"ref n_frames":>14}{"diff":>8}   best-matching ref key')
for k in sorted(cn, key=lambda s: int(s.split('_')[-1])):
    d = rn[k] - cn[k] if k in rn else None
    best = min(rn, key=lambda j: abs(rn[j] - cn[k]))
    mapping[k] = best
    ok = d is not None and 0 <= d <= 40
    bad += (not ok)
    print(f'{k:<12}{cn[k]:>12}{rn.get(k,-1):>14}{d if d is not None else "NA":>8}   {best}{"" if ok else "   <-- MISMATCH"}')
print(f'\n{len(cn)-bad} OK, {bad} mismatched')
if len(set(mapping.values())) < len(mapping):
    print('NOTE: best-match mapping is not one-to-one; mapping may be unreliable')
if out:
    json.dump(mapping, open(out, 'w'), indent=1); print('mapping written to', out)
