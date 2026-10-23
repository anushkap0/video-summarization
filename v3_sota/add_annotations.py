import sys, h5py, numpy as np
clip_path, ref_path = sys.argv[1], sys.argv[2]
ok = bad = 0
with h5py.File(clip_path, 'a') as clip, h5py.File(ref_path, 'r') as ref:
    low = {k.lower(): k for k in ref.keys()}
    for name in list(clip.keys()):
        rk = name if name in ref else low.get(name.lower())
        if rk is None:
            print(f'  WARNING: {name} not in reference'); bad += 1; continue
        g, r = clip[name], ref[rk]
        picks = g['picks'][:]
        rp = r['picks'][:] if 'picks' in r else None
        for f in ('gtscore','gtsummary','user_summary','change_points','n_frame_per_seg'):
            if f not in r: continue
            d = r[f][:]
            if f in ('gtscore','gtsummary') and rp is not None and len(rp) == len(d):
                d = d[np.abs(rp[None,:] - picks[:,None]).argmin(axis=1)]
            if f in g: del g[f]
            g.create_dataset(f, data=d)
        if 'n_frames' in r:
            if 'n_frames' in g: del g['n_frames']
            g.create_dataset('n_frames', data=np.int32(r['n_frames'][()]))
        assert len(g['gtscore']) == len(g['features']), name
        ok += 1
print(f'{clip_path}: annotated {ok} videos, {bad} problems')
