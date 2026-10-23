# Video Summarization on SumMe and TVSum

Supervised video summarization with a Transformer frame scorer, trained with 5-fold cross-validation on SumMe (25 videos) and TVSum (50 videos). Three versions, from a GoogLeNet baseline to a windowed Transformer on CLIP ViT-L/14 features.

| Folder | What it is |
|---|---|
| `shared/` | Dataset loader, DPP frame selection, F1 evaluation, feature check. Copy into the version you run. |
| `v1_baseline/` | GoogLeNet pool5 features (1024-d), 2-layer Transformer scorer. |
| `v2_enhanced/` | 4-layer Pre-LN Transformer, rank loss, optional CLIP ViT-B/32 features. |
| `v3_sota/` | Hierarchical windowed attention + multi-scale conv, CLIP ViT-L/14 or ViT-B/32 or GoogLeNet, test-time augmentation. |

## Method Summary

- **Features.** v1 and the GoogLeNet variant of v3 use pool5 GoogLeNet features [8] (1024-d, from the standard DR-DSN/VASNet HDF5 files [4, 7]). The CLIP variant of v3 uses per-frame CLIP ViT-L/14 image embeddings [3] (768-d, L2-normalised), extracted at 2 fps.
- **Scorer (`TransformerScorerV3`).** Linear input projection, windowed self-attention over local frame neighbourhoods, a learnable CLS token attending globally over the whole video, a learned residual gate that blends the local and global streams, multi-scale 1-D convolutions (kernels 1, 3, 15, 63; cf. [9]) and a sigmoid scoring head. Built on the Transformer encoder [2].
- **Loss.** Mean-squared error against smoothed annotator importance scores, plus a pairwise rank loss, plus (CLIP variant only) a small InfoNCE-style contrastive term [10] that compares CLIP text and frame embeddings. The text is a generic prompt built from the video key (for example, "a highlight summary frame from video 7"), **not** the video title.
- **Summary selection.** The top 15% of frames, chosen either by top-k on the scores or by greedy MAP inference in a quality-diversity determinantal point process (DPP) [11, 5].
- **Evaluation.** 5-fold cross-validation; F-score computed against the user summaries, taking the maximum over annotators.

## Results

5-fold cross-validation, mean F1 over held-out videos, **max-F1 over annotators**, **frame-level selection of 15% of frames** (not shot-level knapsack).

| Model | Selection | SumMe | TVSum |
|---|---|---:|---:|
| v1, GoogLeNet | DPP | 0.369 | 0.320 |
| v3, GoogLeNet (default config, 40 epochs) | DPP | 0.337 ± 0.048 | 0.338 ± 0.033 |
| v3, CLIP ViT-L/14 (small config, 150 epochs) | DPP | 0.361 ± 0.061 | 0.340 ± 0.031 |
| v3, CLIP ViT-L/14 (small config, 150 epochs) | top-k | **0.417 ± 0.062** | **0.394 ± 0.046** |

Same CLIP model with test-time augmentation on held-out videos: DPP 0.350 / 0.341, top-k 0.419 / 0.395. A second configuration (200 epochs, window 128, LR 8e-5) scored within noise of the first (DPP 0.355 / 0.331, top-k 0.410 / 0.390), so the simpler configuration is the one kept in `config_v3.py`.

### How to Read These Numbers

- **Do not compare them directly to published results** (DR-DSN, VASNet, PGL-SUM, DSNet). Those papers select whole shots with a 0/1 knapsack under a 15% length limit, and TVSum is scored with the *average* F1 over annotators. This code selects individual frames and takes the *max* over annotators, so absolute values are not comparable.
- Fold-to-fold spread is large (±0.03 on TVSum, ±0.06 on SumMe). Differences between configurations of about 0.02 are noise.
- Per-fold checkpoints are chosen by F1 on the held-out videos themselves, and the hyperparameters were adjusted over a few runs on the same folds. Both make the numbers slightly optimistic (roughly 0.015–0.02 from checkpoint selection, judging by best-versus-final epoch in the logs).
- Top-k beats DPP by about 0.05 on both datasets here: DPP's diversity step does not help under this metric.
- SumMe `video_21` is truncated in the CLIP features (a video decode error stopped extraction at 7,906 of 9,672 frames).

## Setup

```bash
pip install torch torchvision h5py numpy scipy tqdm opencv-python

# For CLIP features from raw video:
pip install git+https://github.com/openai/CLIP.git "imageio[ffmpeg]" av
```

### Data

Data is **not** included in this repository.

1. **Reference HDF5 files** (GoogLeNet features and annotations in the standard DR-DSN/VASNet format), saved as `data/summe.h5` and `data/tvsum.h5`:
   - [DR-DSN repository](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) [4]
   - [PGL-SUM repository](https://github.com/e-apostolidis/PGL-SUM) [12]

2. **Raw videos** (only needed for CLIP features): SumMe [6] and TVSum [5] originals are available in the [Zenodo dataset archive](https://zenodo.org/records/4884870) (`datasets.tar`, 3.2 GB, CC BY 4.0).
   - [TVSum original source](https://github.com/yalesong/tvsum)
   - [SumMe original source](https://gyglim.github.io/me/vsum/index.html)

## Running

### v1 Baseline

```bash
cp shared/*.py v1_baseline/ && cd v1_baseline
python run_all.py --device cuda
```

### v3 with GoogLeNet Features (No Raw Videos)

```bash
cp shared/*.py v3_sota/ && cd v3_sota
sed -i "s/^BACKBONE = .*/BACKBONE = 'googlenet'/" config_v3.py
python main_v3.py --device cuda --epochs 40
```

### v3 with CLIP ViT-L/14 (Requires Raw Videos)

1. Put the videos in `raw_videos/summe/` and `raw_videos/tvsum/`, named `video_1.mp4`, `video_2.mp4`, and so on. **The numbering must match the keys in the reference HDF5.** Sorted or download order usually does not.

2. Extract features:

   ```bash
   sed -i "s/^BACKBONE = .*/BACKBONE = 'vitl14'/" config_v3.py
   python main_v3.py --extract --device cuda --epochs 150
   ```

   This writes `data/summe_clip_l14.h5` and `data/tvsum_clip_l14.h5`.

3. **Attach the labels to the CLIP files.** Extraction only copies labels if the reference file exists at `data/<name>.h5` at that time. Check and fix the video-to-label pairing:

   ```bash
   python check_mapping.py data/tvsum_clip_l14.h5 data/tvsum.h5
   python remap_annotate.py data/tvsum_clip_l14.h5 data/tvsum.h5 --dry
   python remap_annotate.py data/tvsum_clip_l14.h5 data/tvsum.h5
   python add_annotations.py data/summe_clip_l14.h5 data/summe.h5
   ```

   `remap_annotate.py` pairs each video with its label entry using frame count plus the similarity of the CLIP and GoogLeNet temporal self-similarity structure, then solves a one-to-one assignment. Without this step, TVSum trains on labels belonging to other videos and scores below the random baseline.

4. Train:

   ```bash
   python main_v3.py --device cuda --epochs 150
   ```

Results are written to `results_v3/`.

## Changes from the Original Student Package

- Added the missing `dpp_select` to `shared/dpp_sampler.py` (v2 and v3 failed to import).
- `v3_sota/train_v3.py` and `model_v3.py` imported `config_v2`; they now use `config_v3` (added `MODE`, `get_dataset_path`, and a `backbone` key in the results).
- Model size, dropout, window, warmup, and loss weights in `config_v3.py` were previously ignored (hardcoded defaults). They are now passed to the model and loss.
- **Fixed a data leak in the fold ensemble:** every video was scored by all five fold models, four of which had trained on it, which inflated F1 (SumMe 0.62 instead of 0.34). Each video is now scored only by the model that held it out.
- Added `check_mapping.py`, `remap_annotate.py`, and `add_annotations.py` for building correctly labelled CLIP feature files.

## Known Limitations

- Frame-level selection and max-F1 scoring, as described above.
- Checkpoint selection by held-out F1; no separate validation split.
- Small datasets: 25 and 50 videos.
- No shot-level knapsack selection.
- Only v1 to v3 with top-k or DPP selection are implemented here. This repository does **not** contain code for clustering, optical-flow, C3D or reinforcement-learning baselines, for zero-shot CLIP scoring, or for VSUMM experiments.

## References

[1] Zhang K, Chao WL, Sha F, Grauman K. Video summarization with long short-term memory. *European Conference on Computer Vision (ECCV)*, 2016, pp. 766–782. [DOI](https://doi.org/10.1007/978-3-319-46478-7_47)

[2] Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, et al. Attention is all you need. *Advances in Neural Information Processing Systems 30*, 2017.

[3] Radford A, Kim JW, Hallacy C, Ramesh A, Goh G, Agarwal S, et al. Learning transferable visual models from natural language supervision. *International Conference on Machine Learning (ICML)*, 2021, pp. 8748–8763. [CLIP implementation](https://github.com/openai/CLIP)

[4] Zhou K, Qiao Y, Xiang T. Deep reinforcement learning for unsupervised video summarization with diversity-representativeness reward. *AAAI Conference on Artificial Intelligence 32*(1), 2018. [DOI](https://doi.org/10.1609/aaai.v32i1.12255) · [Code and HDF5 data](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce)

[5] Song Y, Vallmitjana J, Stent A, Jaimes A. TVSum: Summarizing web videos using titles. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 5179–5187.

[6] Gygli M, Grabner H, Riemenschneider H, Van Gool L. Creating summaries from user videos. *European Conference on Computer Vision (ECCV)*, 2014, pp. 505–520. [DOI](https://doi.org/10.1007/978-3-319-10584-0_33)

[7] Fajtl J, Sokeh HS, Argyriou V, Monekosso D, Remagnino P. Summarizing videos with attention (VASNet). *Asian Conference on Computer Vision Workshops*, 2019, pp. 39–54. [DOI](https://doi.org/10.1007/978-3-030-21074-8_4)

[8] Szegedy C, Liu W, Jia Y, Sermanet P, Reed S, Anguelov D, et al. Going deeper with convolutions (GoogLeNet). *IEEE CVPR*, 2015, pp. 1–9. [DOI](https://doi.org/10.1109/CVPR.2015.7298594)

[9] Bai S, Kolter JZ, Koltun V. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*, 2018.

[10] Chen T, Kornblith A, Norouzi M, Hinton G. A simple framework for contrastive learning of visual representations. *ICML*, 2020, pp. 1597–1607.

[11] Kulesza A, Taskar B. Determinantal point processes for machine learning. *Foundations and Trends in Machine Learning 5*(2–3), 2012, pp. 123–286.

[12] Apostolidis E, Balaouras G, Mezaris V, Patras I. Combining global and local attention with positional encoding for video summarization (PGL-SUM). *IEEE International Symposium on Multimedia (ISM)*, 2021. [Code](https://github.com/e-apostolidis/PGL-SUM)

### Related Work (Not Implemented Here)

- Ji Z, Xiong K, Pang Y, Li X. Video summarization with attention-based encoder-decoder networks. *IEEE Transactions on Circuits and Systems for Video Technology 30*(6), 2020, pp. 1709–1717. [DOI](https://doi.org/10.1109/TCSVT.2019.2904899)
- Zhu W, Lu J, Li J, Zhou J. DSNet: A flexible detect-to-summarize network for video summarization. *IEEE Transactions on Image Processing 30*, 2021, pp. 948–962. [DOI](https://doi.org/10.1109/TIP.2020.3039886)
- Apostolidis E, Adamantidou E, Metsai AI, Mezaris V, Patras I. Video summarization using deep neural networks: A survey. *Proceedings of the IEEE 109*(11), 2021, pp. 1838–1863.
