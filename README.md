# Hierarchical Visual-Semantic Feature Learning for Video Summarization with CLIP Transformers

A supervised video summarization project that combines **CLIP visual-semantic representations** with a hierarchical Transformer frame scorer to identify important moments in videos. The project explores multiple model architectures, temporal attention mechanisms, multi-scale feature learning, and summary selection strategies on the SumMe and TVSum benchmarks.

The implementation includes three model versions, progressing from a GoogLeNet baseline to a hierarchical, windowed Transformer using CLIP ViT-L/14 features.

## Overview

- **Datasets:** SumMe (25 videos) and TVSum (50 videos).
- **Architecture:** Hierarchical Transformer frame scorer (`TransformerScorerV3`).
- **Visual features:** GoogLeNet pool5 and CLIP ViT-L/14 or ViT-B/32 embeddings.
- **Training:** Supervised learning with 5-fold cross-validation.
- **Temporal modelling:** Local windowed attention, global video context, and multi-scale temporal convolutions.
- **Loss functions:** Regression loss, pairwise ranking loss, and an optional contrastive objective.
- **Summary generation:** Top-k frame selection or determinantal point process (DPP) selection.
- **Evaluation:** Frame-level F1 with a 15% frame-selection budget and maximum F1 over annotators.

## Repository Structure

| Directory | Description |
|---|---|
| `shared/` | Shared dataset loader, DPP sampler, F1 evaluation, and feature utilities. |
| `v1_baseline/` | Baseline model using 1024-dimensional GoogLeNet pool5 features and a 2-layer Transformer scorer. |
| `v2_enhanced/` | Enhanced 4-layer Pre-LN Transformer with ranking loss and optional CLIP ViT-B/32 features. |
| `v3_sota/` | Hierarchical windowed attention, multi-scale temporal convolutions, CLIP support, and test-time augmentation. |

## Methodology

### 1. Visual Feature Extraction

The project supports two feature extraction approaches.

| Feature extractor | Feature dimension | Usage |
|---|---:|---|
| GoogLeNet pool5 | 1024 | Baseline and compatible v3 experiments using reference HDF5 features. |
| CLIP ViT-L/14 | 768 | Visual-semantic frame embeddings extracted at 2 FPS. |
| CLIP ViT-B/32 | Configuration-dependent | Optional CLIP-based experiments. |

GoogLeNet features are loaded from reference HDF5 files in the standard DR-DSN/VASNet format. CLIP embeddings provide pretrained visual representations that can be used to model semantic content across video frames.

CLIP features are L2-normalised before being passed to the scoring model.

### 2. Hierarchical Transformer Scorer

The `TransformerScorerV3` model combines local temporal reasoning with global video context.

Its main components are:

- **Linear input projection:** Projects input frame features into the model's hidden representation.
- **Windowed self-attention:** Models relationships between frames within local temporal neighbourhoods.
- **Global CLS token:** Learns a video-level representation by attending to the entire sequence.
- **Learnable residual gate:** Blends local and global feature streams.
- **Multi-scale temporal convolutions:** Uses kernel sizes 1, 3, 15, and 63 to capture temporal patterns at different scales.
- **Sigmoid scoring head:** Produces an importance score for each frame.

The architecture combines Transformer-based attention with multi-scale convolutional sequence modelling.

### 3. Training Objectives

The training objective combines complementary learning signals:

- **Mean-squared error (MSE):** Matches predicted importance scores to smoothed human-annotated importance scores.
- **Pairwise ranking loss:** Encourages more important frames to receive higher predicted scores than less important frames.
- **Contrastive loss:** The CLIP variant optionally uses a small InfoNCE-style objective to compare frame embeddings with text embeddings.

The contrastive objective uses generic prompts derived from video keys, such as `a highlight summary frame from video 7`. These prompts are not the original video titles.

### 4. Summary Generation

The model converts frame importance scores into a compact video summary using one of two strategies.

| Strategy | Description |
|---|---|
| Top-k | Selects the highest-scoring frames within the frame-selection budget. |
| DPP | Uses greedy maximum a posteriori inference with a determinantal point process to balance frame quality and diversity. |

Both strategies use a 15% frame-selection budget in the reported experiments. This is **frame-level selection**, not standard shot-level selection using a knapsack algorithm.

### 5. Evaluation Protocol

The project uses 5-fold cross-validation on SumMe and TVSum. Each fold evaluates held-out videos, and the reported F1 uses the maximum score across human annotators.

The evaluation protocol differs from many published video summarization benchmarks, which use shot-level selection and average F1 across annotators for TVSum.

## Experimental Results

The following results are from the current implementation and should be interpreted under its specific frame-level evaluation protocol.

### Main Results

| Model | Selection | SumMe F1 | TVSum F1 |
|---|---|---:|---:|
| v1 — GoogLeNet | DPP | 0.369 | 0.320 |
| v3 — GoogLeNet, 40 epochs | DPP | 0.337 ± 0.048 | 0.338 ± 0.033 |
| v3 — CLIP ViT-L/14, 150 epochs | DPP | 0.361 ± 0.061 | 0.340 ± 0.031 |
| v3 — CLIP ViT-L/14, 150 epochs | Top-k | **0.417 ± 0.062** | **0.394 ± 0.046** |

### Test-Time Augmentation

| Model configuration | Selection | SumMe F1 | TVSum F1 |
|---|---|---:|---:|
| CLIP ViT-L/14 with TTA | DPP | 0.350 | 0.341 |
| CLIP ViT-L/14 with TTA | Top-k | 0.419 | 0.395 |

A second CLIP configuration using 200 epochs, window size 128, and a learning rate of `8e-5` produced DPP scores of 0.355 on SumMe and 0.331 on TVSum, and top-k scores of 0.410 and 0.390, respectively.

The simpler configuration is retained in `config_v3.py`.

### Interpretation

- Top-k selection performs better than DPP under the current evaluation protocol, by approximately 0.05 F1 on both datasets in the primary CLIP experiment.
- Fold-to-fold variation is substantial, particularly on SumMe.
- Differences of approximately 0.02 between configurations may reflect experimental variation rather than a reliable improvement.
- Checkpoint selection uses held-out F1, and hyperparameters were adjusted across multiple runs on the same folds. Consequently, the results may be optimistic.
- SumMe `video_21` has incomplete CLIP features because video decoding stopped at frame 7,906 of 9,672.

**Benchmark caution:** These scores should not be compared directly with published DR-DSN, VASNet, PGL-SUM, or DSNet results. The evaluation protocol differs in both frame-versus-shot selection and annotator aggregation.

## Installation

Install the base dependencies:

```bash
pip install torch torchvision h5py numpy scipy tqdm opencv-python
```

For CLIP feature extraction from raw videos, install:

```bash
pip install git+https://github.com/openai/CLIP.git "imageio[ffmpeg]" av
```

A CUDA-enabled PyTorch installation is recommended for GPU training. Install the PyTorch build compatible with your CUDA environment if needed.

## Dataset Preparation

The datasets and extracted features are **not included** in this repository.

### Reference HDF5 Files

Obtain reference HDF5 files containing GoogLeNet features and annotations in the standard DR-DSN/VASNet format.

Save them as:

```text
data/
├── summe.h5
└── tvsum.h5
```

Sources:

- [DR-DSN — Reference implementation and data](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce)
- [PGL-SUM — Reference implementation](https://github.com/e-apostolidis/PGL-SUM)

### Raw Videos for CLIP

Raw videos are required only when extracting CLIP features.

- [SumMe and TVSum dataset archive on Zenodo](https://zenodo.org/records/4884870)
- [TVSum original repository](https://github.com/yalesong/tvsum)
- [SumMe original dataset page](https://gyglim.github.io/me/vsum/index.html)

Organise the videos as follows:

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

**Important:** Video numbering must match the keys in the reference HDF5 files. Alphabetical ordering or download order may not preserve the correct correspondence between videos and annotations.

## Running Experiments

### v1 — GoogLeNet Baseline

From the repository root:

```bash
cp shared/*.py v1_baseline/
cd v1_baseline
python run_all.py --device cuda
```

### v3 — GoogLeNet Features

This configuration does not require raw videos if the reference HDF5 files are available.

```bash
cp shared/*.py v3_sota/
cd v3_sota
sed -i "s/^BACKBONE = .*/BACKBONE = 'googlenet'/" config_v3.py
python main_v3.py --device cuda --epochs 40
```

### v3 — CLIP ViT-L/14

#### Step 1: Configure the backbone

Run from `v3_sota/`:

```bash
sed -i "s/^BACKBONE = .*/BACKBONE = 'vitl14'/" config_v3.py
```

#### Step 2: Extract CLIP features

Ensure that the raw videos and reference HDF5 files are available, then run:

```bash
python main_v3.py --extract --device cuda --epochs 150
```

The expected feature files are:

```text
data/
├── summe_clip_l14.h5
└── tvsum_clip_l14.h5
```

Actual output paths depend on the configured dataset paths.

#### Step 3: Verify annotation mapping

Feature extraction may not correctly attach labels if reference files are missing or video numbering differs.

First, inspect the TVSum mapping:

```bash
python check_mapping.py data/tvsum_clip_l14.h5 data/tvsum.h5
```

Preview the proposed remapping:

```bash
python remap_annotate.py data/tvsum_clip_l14.h5 data/tvsum.h5 --dry
```

After inspecting the proposed assignments, apply the remapping:

```bash
python remap_annotate.py data/tvsum_clip_l14.h5 data/tvsum.h5
```

Attach the SumMe annotations:

```bash
python add_annotations.py data/summe_clip_l14.h5 data/summe.h5
```

The TVSum mapping utility uses frame counts and similarities between CLIP and GoogLeNet temporal self-similarity structures to infer a one-to-one video-to-annotation assignment.

Incorrect annotation mapping can cause the model to train on labels belonging to different videos and produce misleading results.

#### Step 4: Train the model

```bash
python main_v3.py --device cuda --epochs 150
```

Results are written to `results_v3/`.

## Implementation Improvements

The repository includes several fixes to the original student package:

- Added the missing `dpp_select` implementation to `shared/dpp_sampler.py`.
- Fixed v3 imports to use `config_v3` rather than `config_v2`.
- Added configuration support for `MODE`, `get_dataset_path`, and backbone tracking in experiment results.
- Corrected configuration propagation for model size, dropout, window size, warmup, and loss weights.
- Fixed fold-ensemble data leakage: each held-out video is now evaluated only by the model that excluded it during training.
- Added utilities for inspecting and repairing CLIP feature-to-annotation mappings: `check_mapping.py`, `remap_annotate.py`, and `add_annotations.py`.

The fold-ensemble fix is particularly important. Previously, videos were scored by all five fold models, including models that had trained on those videos. This inflated SumMe F1 to approximately 0.62, compared with approximately 0.34 after correcting the evaluation.

## Limitations and Future Work

- The current evaluation selects individual frames rather than complete shots.
- The implementation uses maximum F1 over annotators rather than the average-F1 protocol commonly used for TVSum.
- There is no separate validation split; checkpoint selection uses held-out fold performance.
- The datasets are small, containing 25 SumMe videos and 50 TVSum videos.
- Hyperparameter tuning across the same folds can introduce selection bias.
- SumMe `video_21` has incomplete CLIP features.
- The repository does not implement clustering, optical-flow, C3D, or reinforcement-learning baselines, zero-shot CLIP scoring, or VSUMM experiments.

Potential future extensions include independent validation, standard shot-level knapsack evaluation, repeated cross-validation, stronger video-text alignment, and comparison with published baselines under identical evaluation protocols.

## References

1. Zhang K, Chao WL, Sha F, Grauman K. Video summarization with long short-term memory. *European Conference on Computer Vision (ECCV)*, 2016, pp. 766–782. [DOI](https://doi.org/10.1007/978-3-319-46478-7_47)

2. Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, et al. Attention is all you need. *Advances in Neural Information Processing Systems 30*, 2017.

3. Radford A, Kim JW, Hallacy C, Ramesh A, Goh G, Agarwal S, et al. Learning transferable visual models from natural language supervision. *International Conference on Machine Learning (ICML)*, 2021, pp. 8748–8763. [CLIP code](https://github.com/openai/CLIP)

4. Zhou K, Qiao Y, Xiang T. Deep reinforcement learning for unsupervised video summarization with diversity-representativeness reward. *AAAI Conference on Artificial Intelligence 32*(1), 2018. [DOI](https://doi.org/10.1609/aaai.v32i1.12255) · [Code and HDF5 data](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce)

5. Song Y, Vallmitjana J, Stent A, Jaimes A. TVSum: Summarizing web videos using titles. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 5179–5187.

6. Gygli M, Grabner H, Riemenschneider H, Van Gool L. Creating summaries from user videos. *European Conference on Computer Vision (ECCV)*, 2014, pp. 505–520. [DOI](https://doi.org/10.1007/978-3-319-10584-0_33)

7. Fajtl J, Sokeh HS, Argyriou V, Monekosso D, Remagnino P. Summarizing videos with attention (VASNet). *Asian Conference on Computer Vision Workshops*, 2019, pp. 39–54. [DOI](https://doi.org/10.1007/978-3-030-21074-8_4)

8. Szegedy C, Liu W, Jia Y, Sermanet P, Reed S, Anguelov A, et al. Going deeper with convolutions (GoogLeNet). *IEEE CVPR*, 2015, pp. 1–9. [DOI](https://doi.org/10.1109/CVPR.2015.7298594)

9. Bai S, Kolter JZ, Koltun V. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv:1803.01271*, 2018.

10. Chen T, Kornblith A, Norouzi M, Hinton G. A simple framework for contrastive learning of visual representations. *ICML*, 2020, pp. 1597–1607.

11. Kulesza A, Taskar B. Determinantal point processes for machine learning. *Foundations and Trends in Machine Learning 5*(2–3), 2012, pp. 123–286.

12. Apostolidis E, Balaouras G, Mezaris V, Patras I. Combining global and local attention with positional encoding for video summarization (PGL-SUM). *IEEE International Symposium on Multimedia (ISM)*, 2021. [Code](https://github.com/e-apostolidis/PGL-SUM)

### Related Work

The following methods are relevant to video summarization but are not implemented in this repository:

- Ji Z, Xiong K, Pang Y, Li X. Video summarization with attention-based encoder-decoder networks. *IEEE Transactions on Circuits and Systems for Video Technology 30*(6), 2020, pp. 1709–1717. [DOI](https://doi.org/10.1109/TCSVT.2019.2904899)
- Zhu W, Lu J, Li J, Zhou J. DSNet: A flexible detect-to-summarize network for video summarization. *IEEE Transactions on Image Processing 30*, 2021, pp. 948–962. [DOI](https://doi.org/10.1109/TIP.2020.3039886)
- Apostolidis E, Adamantidou E, Metsai AI, Mezaris V, Patras I. Video summarization using deep neural networks: A survey. *Proceedings of the IEEE 109*(11), 2021, pp. 1838–1863. [DOI](https://doi.org/10.1109/JPROC.2021.3100755)
