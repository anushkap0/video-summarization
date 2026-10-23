import sys, json, h5py, numpy as np
from scipy.optimize import linear_sum_assignment

clip_path, ref_path = sys.argv[1], sys.argv[2]
dry = '--dry' in sys.argv
B = 64
TOL = 60

def ssm_vec(F):
    F = np.asarray(F, 'float64')
    if len(F) < B:
        F = F[np.linspace(0, len(F) - 1, B).round().astype(int)]
    idx = np.array_split(np.arange(len(F)), B)
    P = np.stack([F[i].mean(0) for i in idx])
    P = P - P.mean(0, keepdims=True)
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    S = P @ P.T
    v = S[np.triu_indices(B, 1)]
    return (v - v.mean()) / (v.std() + 1e-8)

with h5py.File(clip_path) as c, h5py.File(ref_path) as r:
    ck = sorted(c.keys(), key=lambda s: int(s.split('_')[-1]))
    rk = sorted(r.keys(), key=lambda s: int(s.split('_')[-1]))
    cn = {k: int(c[k]['picks'][-1]) + 1 for k in ck}
    rn = {k: int(r[k]['n_frames'][()]) for k in rk}
    cv = {k: ssm_vec(c[k]['features'][:]) for k in ck}
    rv = {k: ssm_vec(r[k]['features'][:]) for k in rk}

n = len(ck)
corr = np.full((n, len(rk)), -1.0)
ok = np.zeros((n, len(rk)), bool)
for i, a in enumerate(ck):
    for j, b in enumerate(rk):
        d = rn[b] - cn[a]
        if 0 <= d <= TOL:
            ok[i, j] = True
        corr[i, j] = float(np.mean(cv[a] * rv[b]))
cost = -corr + 10.0 * (~ok)
ri, ci = linear_sum_assignment(cost)
mapping = {ck[i]: rk[j] for i, j in zip(ri, ci)}

print(f'{"clip key":<10}{"-> ref key":<12}{"len ok":>8}{"struct corr":>13}{"2nd best":>10}')
weak = 0
for i, j in zip(ri, ci):
    others = np.sort(np.where(ok[i], corr[i], -1))[::-1]
    second = others[1] if len(others) > 1 else -1
    flag = '' if ok[i, j] and corr[i, j] > 0.3 else '   <-- CHECK'
    weak += bool(flag)
    print(f'{ck[i]:<10}{rk[j]:<12}{str(bool(ok[i,j])):>8}{corr[i,j]:>13.3f}{second:>10.3f}{flag}')
print(f'\nidentity-mapped: {sum(k==v for k,v in mapping.items())}/{n};  weak/unverified: {weak}')
json.dump(mapping, open(clip_path + '.map.json', 'w'), indent=1)
print('mapping saved to', clip_path + '.map.json')
if dry:
    print('--dry: annotations NOT written'); sys.exit()

with h5py.File(clip_path, 'a') as c, h5py.File(ref_path) as r:
    for a, b in mapping.items():
        g, s = c[a], r[b]
        picks = g['picks'][:]; rp = s['picks'][:]
        for f in ('gtscore', 'gtsummary', 'user_summary', 'change_points', 'n_frame_per_seg'):
            if f not in s: continue
            d = s[f][:]
            if f in ('gtscore', 'gtsummary') and len(rp) == len(d):
                d = d[np.abs(rp[None, :] - picks[:, None]).argmin(axis=1)]
            if f in g: del g[f]
            g.create_dataset(f, data=d)
        if 'n_frames' in g: del g['n_frames']
        g.create_dataset('n_frames', data=np.int32(s['n_frames'][()]))
        g.attrs['label_key'] = b
print('annotations rewritten.')
