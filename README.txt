# Video Summarization on SumMe and TVSum

Supervised video summarization with a Transformer frame scorer, trained with 5-fold cross-validation on SumMe (25 videos) and TVSum (50 videos). Three versions, from a GoogLeNet baseline to a windowed Transformer on CLIP ViT-L/14 features.

## Repository Structure

| Folder | What it is |
|---|---|
| `shared/` | Dataset loader, DPP frame selection, F1 evaluation, and feature checks. Copy into the version you run. |
| `v1_baseline/` | GoogLeNet pool5 features (1024-d), 2-layer Transformer scorer. |
| `v2_enhanced/` | 4-layer Pre-LN Transformer, rank loss, optional CLIP ViT-B/32 features. |
| `v3_sota/` | Hierarchical windowed attention, multi-scale convolutions, CLIP ViT-L/14 or ViT-B/32 or GoogLeNet, and test-time augmentation. |

## Method Summary

- **Features:** v1 and the GoogLeNet variant of v3 use pool5 GoogLeNet features [8] (1024-d, from standard DR-DSN/VASNet HDF5 files [4, 7]). The CLIP variant of v3 uses per-frame CLIP ViT-L/14 embeddings [3] (768-d, L2-normalised), extracted at 2 FPS.
- **Scorer (`TransformerScorerV3`):** Linear input projection, windowed self-attention over local frame neighbourhoods, a learnable CLS token for global context, a learned residual gate blending local and global streams, multi-scale 1-D convolutions (kernels 1, 3, 15, 63) [9], and a sigmoid scoring head based on the Transformer encoder [2].
- **Loss:** Mean-squared error against smoothed annotator importance scores, pairwise ranking loss, and (CLIP variant only) a small InfoNCE-style contrastive term [10] comparing CLIP text and frame embeddings. The text is a generic prompt derived from the video key, such as "a highlight summary frame from video 7", not the video title.
- **Summary selection:** The top 15% of frames, selected using top-k scores or greedy MAP inference in a quality-diversity determinantal point process (DPP) [11, 5].
- **Evaluation:** 5-fold cross-validation, with F1 computed against user summaries using the maximum score over annotators.

## Results

5-fold cross-validation, mean F1 over held-out videos, **max-F1 over annotators**, and **frame-level selection of 15% of frames** (not shot-level knapsack).

| Model | Selection | SumMe | TVSum |
|---|---|---:|---:|
| v1, GoogLeNet | DPP | 0.369 | 0.320 |
| v3, GoogLeNet (default config, 40 epochs) | DPP | 0.337 ± 0.048 | 0.338 ± 0.033 |
| v3, CLIP ViT-L/14 (small config, 150 epochs) | DPP | 0.361 ± 0.061 | 0.340 ± 0.031 |
| v3, CLIP ViT-L/14 (small config, 150 epochs) | Top-k | **0.417 ± 0.062** | **0.394 ± 0.046** |

### Test-Time Augmentation

| Configuration | Selection | SumMe F1 | TVSum F1 |
|---|---|---:|---:|
| CLIP ViT-L/14 with TTA | DPP | 0.350 | 0.341 |
| CLIP ViT-L/14 with TTA | Top-k | 0.419 | 0.395 |

A second configuration (200 epochs, window 128, learning rate `8e-5`) scored within experimental variation of the first: DPP 0.355 / 0.331 and top-k 0.410 / 0.390. The simpler configuration is retained in `config_v3.py`.

### How to Read These Numbers

- **Do not compare these results directly with published benchmarks** such as DR-DSN, VASNet, PGL-SUM, and DSNet. Those methods generally select whole shots using a 0/1 knapsack under a 15% length limit, and TVSum is commonly evaluated using average F1 over annotators. This implementation selects individual frames and uses maximum F1 over annotators.
- Fold-to-fold variation is substantial: approximately ±0.03 on TVSum and ±0.06 on SumMe. Differences of around 0.02 between configurations may be noise.
- Checkpoints are selected using F1 on held-out videos, and hyperparameters were adjusted across runs on the same folds. This makes the reported results potentially optimistic; logs suggest a checkpoint-selection effect of approximately 0.015–0.020.
- Top-k outperforms DPP by approximately 0.05 on both datasets in the primary CLIP experiment.
- SumMe `video_21` is truncated in the CLIP features: a decoding error stopped extraction at 7,906 of 9,672 frames.

## Setup

Install the base dependencies:

```bash
pip install torch torchvision h5py numpy scipy tqdm opencv-python
```

For CLIP feature extraction from raw video:

```bash
pip install git+https://github.com/openai/CLIP.git "imageio[ffmpeg]" av
```

## Data

Data is **not included** in this repository.

### Reference HDF5 Files

Obtain GoogLeNet features and annotations in the standard DR-DSN/VASNet format and save them as `data/summe.h5` and `data/tvsum.h5`.

- [DR-DSN repository and data](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) [4]
- [PGL-SUM repository](https://github.com/e-apostolidis/PGL-SUM) [12]

### Raw Videos

Raw videos are required only for CLIP feature extraction.

- [Zenodo dataset archive](https://zenodo.org/records/4884870) — `datasets.tar`, approximately 3.2 GB, CC BY 4.0.
- [TVSum original source](https://github.com/yalesong/tvsum) [5]
- [SumMe original source](https://gyglim.github.io/me/vsum/index.html) [6]

Organize the raw videos as follows:

```text
raw_videos/
├── summe/
│   ├── video_1.mp4
│   ├── video_2.mp4
│   └── ...
└── tvsum/
    ├── video_1.mp4
    ├── video_2.mp4
    └── ...
```

**Important:** Video numbering must match the keys in the reference HDF5 files. Sorted or download order usually does not preserve the correct mapping.

## Running

### v1 Baseline

Run from the repository root:

```bash
cp shared/*.py v1_baseline/
cd v1_baseline
python run_all.py --device cuda
```

### v3 with GoogLeNet Features

This experiment does not require raw videos.

```bash
cp shared/*.py v3_sota/
cd v3_sota
sed -i "s/^BACKBONE = .*/BACKBONE = 'googlenet'/" config_v3.py
python main_v3.py --device cuda --epochs 40
```

### v3 with CLIP ViT-L/14

**Step 1: Prepare the raw videos.**

Place the videos in `raw_videos/summe/` and `raw_videos/tvsum/`. Ensure their numbering matches the reference HDF5 keys.

**Step 2: Extract CLIP features.**

Run from `v3_sota/`:

```bash
sed -i "s/^BACKBONE = .*/BACKBONE = 'vitl14'/" config_v3.py
python main_v3.py --extract --device cuda --epochs 150
```

This is expected to generate `data/summe_clip_l14.h5` and `data/tvsum_clip_l14.h5`, depending on the configured dataset paths.

**Step 3: Verify and repair annotation mapping.**

Extraction only copies labels if the reference file exists at `data/<name>.h5` at extraction time.

Check the TVSum mapping:

```bash
python check_mapping.py data/tvsum_clip_l14.h5 data/tvsum.h5
```

Preview the proposed remapping:

```bash
python remap_annotate.py data/tvsum_clip_l14.h5 data/tvsum.h5 --dry
```

Apply the remapping after inspecting the output:

```bash
python remap_annotate.py data/tvsum_clip_l14.h5 data/tvsum.h5
```

Attach SumMe annotations:

```bash
python add_annotations.py data/summe_clip_l14.h5 data/summe.h5
```

The TVSum remapping utility uses frame counts and similarities between CLIP and GoogLeNet temporal self-similarity structures to infer a one-to-one video-to-annotation assignment. Without correct mapping, TVSum may train against labels belonging to other videos and perform below the random baseline.

**Step 4: Train the model.**

```bash
python main_v3.py --device cuda --epochs 150
```

Results are written to `results_v3/`.

## Changes from the Original Student Package

- Added the missing `dpp_select` implementation to `shared/dpp_sampler.py`, fixing import failures in v2 and v3.
- Fixed `v3_sota/train_v3.py` and `model_v3.py` to use `config_v3` instead of `config_v2`. Added `MODE`, `get_dataset_path`, and a `backbone` key in the results.
- Fixed configuration handling so model size, dropout, window size, warmup, and loss weights are passed to the model and loss instead of being ignored by hardcoded defaults.
- **Fixed fold-ensemble data leakage:** Previously, every video was scored by all five fold models, including four models that had trained on that video. This inflated SumMe F1 to approximately 0.62 instead of 0.34. Each video is now scored only by its held-out fold model.
- Added `check_mapping.py`, `remap_annotate.py`, and `add_annotations.py` to build correctly labelled CLIP feature files.

## Known Limitations

- Frame-level selection and max-F1 scoring rather than the standard shot-level protocol.
- Checkpoint selection uses held-out F1; there is no separate validation split.
- Small datasets: 25 SumMe videos and 50 TVSum videos.
- No shot-level knapsack selection.
- Only v1–v3 with top-k or DPP selection are implemented. The repository does not include clustering, optical-flow, C3D, reinforcement-learning baselines, zero-shot CLIP scoring, or VSUMM experiments.

## References

[1] Zhang K, Chao WL, Sha F, Grauman K. Video summarization with long short-term memory. *European Conference on Computer Vision (ECCV)*, 2016, pp. 766–782. [DOI](https://doi.org/10.1007/978-3-319-46478-7_47)

[2] Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, et al. Attention is all you need. *Advances in Neural Information Processing Systems 30*, 2017.

[3] Radford A, Kim JW, Hallacy C, Ramesh A, Goh G, Agarwal S, et al. Learning transferable visual models from natural language supervision. *International Conference on Machine Learning (ICML)*, 2021, pp. 8748–8763. [CLIP code](https://github.com/openai/CLIP)

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
