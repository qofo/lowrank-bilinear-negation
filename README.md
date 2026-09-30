# Analyzing CLIP's Negation Bottleneck under Object-Level Control with a Low-Rank Bilinear Transform

**객체 단위 통제로 본 CLIP 부정 이해의 병목 분석과 저계수 이중선형 변환**

[Junseo Lee](mailto:junseolee@ajou.ac.kr), [Hyunsouk Cho](mailto:hyunsouk@ajou.ac.kr)\*
Department of Software & Computer Engineering, Ajou University

**The 7th Korea Artificial Intelligence Conference (Korea AI 2026)**

[![Korea AI 2026](https://img.shields.io/badge/Korea%20AI-2026-blue.svg)](https://koreaai.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-ee4c2c.svg)

[한국어 문서 (README.ko.md)](README.ko.md)

<p align="center">
  <img src="assets/fig1_minimal_pair.png" width="70%" alt="A minimal pair: the same COCO scene with the cups removed and with the pizza removed, crossed with two captions that use the same words but negate different objects. CLIP scores the mismatched pairs above the matched ones.">
</p>

CLIP fails on negated queries. We ask *where* the failure happens. We fix the scene by pairing two images
made from one photo, one with object O removed and one with another object removed, and cross them with two
captions that contain the same words but negate different objects. This turns negation into a 2×2 decision:
the two matching image–caption pairs must outscore the two mismatched ones.

We then rule out candidate bottlenecks one stage at a time:

- **Representation.** Linear probes recover object presence from image embeddings and negation from text
  embeddings, above a randomly initialised encoder.
- **Alignment.** Rotating the text probe normal onto the image probe normal (cosine 1.000) leaves 2×2 accuracy
  at about one fifth of chance.
- **Similarity function.** Replacing the identity matrix hidden inside cosine similarity with a shared
  low-rank bilinear transform, `S(e_I, e_T) = (U e_I)ᵀ(V e_T)`, improves 2×2 accuracy on objects never seen in
  training and improves negated items on NegBench MCQ, without retraining either encoder.

## News

- **2026.10.01**: Oral presentation (online) at Korea AI 2026.
- **2026.09**: Code, trained matrices and the reference outputs behind every number in the paper are released.

## Method

For an object O, let `T+`/`T−` be a caption that affirms/negates O and `I+`/`I−` an image with/without O.
Writing `S(u, v)` for the score of caption `T_u` with image `I_v`, a scorer is correct on a pair when

```
ΔS = min(S(+,+), S(−,−)) − max(S(−,+), S(+,−)) > 0
```

If the four scores were in random order, this would hold with probability 1/6 (16.67%). Cosine similarity
meets it on only 1.35% of pairs, so the failure is systematic, not noise.

Cosine similarity is the bilinear form `e_Iᵀ W e_T` with `W = I`. We learn one shared low-rank `W = UᵀV` with
`U, V ∈ R^{r×n}` (`2nr` parameters instead of `n²`) using a hinge loss on ΔS. One `W` is shared across all
objects, and it is evaluated on objects held out of training.

## Results

**Table 1.** Per-object linear probing accuracy (mean over objects) on image and text embeddings, CLIP ViT-B/32.

| Encoder | Image | Text |
|:--|:--:|:--:|
| Random init | 0.52 | 0.54 |
| OpenCLIP | 0.66 | 0.86 |
| OpenCLIP (ft) [NegBench] | 0.65 | 0.93 |

**Table 2.** Alignment of per-object probe normals and the resulting 2×2 accuracy.

| Similarity | cos(d_I, R d_T) | ΔS > 0 (%) |
|:--|:--:|:--:|
| Cosine (baseline) | 0.219 | 1.35 |
| Orthogonal rotation R_o | 1.000 | 3.25 |
| Random rotation | 0.002 | 0.12 |

**Figure 2.** 2×2 accuracy of the low-rank bilinear transform against rank, and of LABCLIP.

<p align="center"><img src="assets/fig2_rank_curve.png" width="55%" alt="2x2 accuracy against rank r from 1 to 512. The accuracy on held-out objects rises above the 16.67% chance line and levels off near 33% from r = 16, while the accuracy on training objects keeps rising. LABCLIP, with as many parameters as r = 256, stays below chance even on its training objects."></p>

**Table 3.** External benchmarks (%). Both transforms are trained on our BEAF pairs and evaluated zero-shot.
Ours uses r = 32. Avg includes the mixed (hybrid) items.

| Similarity | COCO Avg | COCO Pos | COCO Neg | VOC2007 Avg | VOC2007 Pos | VOC2007 Neg | CheXpert |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Random | 25.00 | 25.00 | 25.00 | 25.00 | 25.00 | 25.00 | 50.00 |
| Cosine | 39.30 | 69.14 | 6.84 | 38.72 | 82.55 | 3.35 | 55.61 |
| LABCLIP | **47.02** | **82.23** | 6.42 | 40.95 | **84.75** | 6.46 | 37.43 |
| **Ours** | 45.25 | 61.52 | **24.22** | **51.88** | 73.61 | **22.40** | **58.82** |

## Installation

```bash
git clone https://github.com/qofo/lowrank-bilinear-negation.git
cd lowrank-bilinear-negation
conda create -n lbn python=3.9 -y && conda activate lbn
pip install -r requirements.txt
```

The code uses a vendored, modified OpenCLIP (`benchmarks/src/open_clip`). It is not installed as a package.
`scripts/reproduce_paper.sh` puts it on `PYTHONPATH`. To run a module by hand, first set the path from the
repository root:

```bash
export PYTHONPATH="$PWD:$PWD/benchmarks:$PWD/benchmarks/src"
```

## Data

The CSVs are included in `benchmarks/data/images/`. The images come from their original sources:

| Dataset | Used for | Source | Manifest |
|:--|:--|:--|:--|
| BEAF v1 (`beaf_dataset_ver1.zip`, 2,224 images) | Tables 1–2, Figures 1–2, training both transforms | [Google Drive (BEAF authors)](https://drive.google.com/file/d/1Xx7j8Hz8QX3Fl_hpSBet6r15njhwCgeR/view) | `beaf.sha256` |
| COCO 2017 val (`val2017.zip`) | Table 3, COCO MCQ | [cocodataset.org](http://images.cocodataset.org/zips/val2017.zip) | `coco_val2017.sha256` |
| PASCAL VOC2007 test (`VOCtest_06-Nov-2007.tar`) | Table 3, VOC2007 MCQ | [host.robots.ox.ac.uk](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar) | `voc2007.sha256` |
| CheXpert-v1.0-small (validation split) | Table 3, CheXpert | [Kaggle mirror used by NegBench](https://www.kaggle.com/datasets/ashery/chexpert); the CheXpert Research Use Agreement applies | `chexpert.sha256` |

`scripts/prepare_data.sh` puts each archive where the CSVs expect it. It then checks every image the
CSVs reference against the SHA-256 manifests in `benchmarks/data/checksums/`:

```bash
bash scripts/prepare_data.sh --beaf beaf_dataset_ver1.zip --coco_val2017 val2017.zip \
    --voc2007 VOCtest_06-Nov-2007.tar --chexpert /path/to/CheXpert-v1.0-small
```

The BEAF zip alone is enough for everything except Table 3.

**Checkpoints.**
- The OpenAI CLIP ViT-B/32 weights are downloaded automatically by OpenCLIP.
- The fine-tuned encoder of Table 1 is NegBench's `CLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01`, from
  the [NegBench model folder](https://drive.google.com/drive/folders/1kSEq0mkV1t1T8GuOAM65iz_iAA7e5gxB). Place
  it at `benchmarks/models/CLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01/checkpoint.pt`
  (sha256 `23a68e31da76f95670c4167cb2980e52fbd1eba77c8932cc36ade75efadfeb70`).
- The two trained matrices of Table 3 are in `checkpoints/`: Ours (`delta_warmstart_rank32.pt`) and LABCLIP
  (`labclip_official_recipe_bilinear.pt`).

## Reproducing the paper

```bash
bash scripts/reproduce_paper.sh all      # or name stages, e.g. table2 figure2 compare
```

Every stage recomputes its embeddings from the images and writes to `outputs/paper/` (set `OUT` to change it).
The last stage, `compare`, writes `outputs/paper/compare_with_paper.csv`. It puts each number printed in the
paper next to the reference log it came from (`results/reference/`) and the value just reproduced.

On one RTX 4070 the full run takes about 90 minutes, 70 of them for the three Table 1 probes. Run from a fresh
environment, it reproduces all 61 numbers printed in the paper and both trained matrices bit for bit. The one
exception is the random-initialisation row of Table 1. The original run did not seed the random encoder, so its
value moves in the third decimal (0.5246 / 0.5441 against 0.5230 / 0.5354) but still rounds to the printed
0.52 / 0.54. The script now seeds it, so repeated runs agree.

| Paper item | Stage | Population | Script |
|:--|:--|:--|:--|
| Table 1 | `table1_openai`, `table1_random`, `table1_negft` | image: 76 objects, original vs. object-removed image pairs (GroupKFold over pairs); text: 69 objects, word-swapped captions (StratifiedKFold) | `eval_probe_failure_inspector.py` |
| Sec. 2.2 (1.35%), Table 2 | `table2` | 33 objects, 1,357 original vs. object-removed pairs; held-out columns use GroupKFold over source images | `eval_per_object_alignment_intervention.py` |
| Figure 2 (curve) | `figure2` | 42 objects, 2,480 pairs (each object removed from one image, another object from the other); GroupKFold over objects | `eval_single_w_generalization.py` |
| Figure 2 (LABCLIP star) | `labclip`, `star` | LABCLIP trained on all 2,771 pairs (77 objects), scored on the 42 objects | `train_labclip_official_recipe.py`, `score_delta_s_for_w.py` |
| Table 3 (Ours) | `ours` | rank-32 W trained on all 42 objects / 2,480 pairs | `train_and_save_narrow_rank32_w.py` |
| Table 3 | `table3` | COCO MCQ 5,914 items, VOC2007 MCQ 5,031, CheXpert 187 | `eval_w_ckpts_mcq.py` |
| Figures 1–2 | `figure1_plot`, `figure2_plot` | pizza/cup pair of Figure 1 | `paper_figures/` |

To score the released matrices on the external benchmarks without training anything:

```bash
python -m benchmarks.src.evaluation.eval_w_ckpts_mcq \
    --ckpts checkpoints/delta_warmstart_rank32.pt checkpoints/labclip_official_recipe_bilinear.pt \
    --targets benchmarks/data/images/COCO_val_mcq_llama3.1_rephrased.csv \
              benchmarks/data/images/VOC2007_mcq_llama3.1_rephrased.csv \
              benchmarks/data/images/chexpert_binary_mcq_control_valid_only.csv \
    --output outputs/table3_released.json
```

Tests: `python -m pytest benchmarks/tests` (about one minute, no images needed).

## Repository structure

```
benchmarks/
  src/
    open_clip/          vendored OpenCLIP (modified by NegBench)
    analysis/           feature extraction, feature cache, shared config and CLI helpers
    evaluation/         every experiment of the paper (entry points listed above)
    training/           CsvMCQDataset for the MCQ benchmarks
    data_generation/    builds the word-swapped caption pairs from BEAF
  data/images/          CSVs (captions + relative image paths)
  data/checksums/       SHA-256 manifests of every image the CSVs use
  tests/                invariants the results rest on (e.g. the ΔS identity, fold grouping, tie handling)
checkpoints/            trained W of Ours (rank 32) and of LABCLIP
concept_sets/           the 33-object set of Table 2
paper_figures/          Figure 1 and Figure 2
results/reference/      the original outputs every paper number was read from
scripts/                prepare_data.sh, reproduce_paper.sh
```

## Citation

```bibtex
@inproceedings{lee2026negationbottleneck,
  title     = {Analyzing {CLIP}'s Negation Bottleneck under Object-Level Control with a Low-Rank Bilinear Transform},
  author    = {Lee, Junseo and Cho, Hyunsouk},
  booktitle = {The 7th Korea Artificial Intelligence Conference (Korea AI 2026)},
  year      = {2026}
}
```

## Acknowledgements

This code builds on [NegBench](https://github.com/m1k2zoo/negbench), which provides the code base, the MCQ
benchmarks and the fine-tuned encoder, and on [OpenCLIP](https://github.com/mlfoundations/open_clip). The
controlled image pairs come from [BEAF](https://github.com/postech-ami/BEAF). The LABCLIP baseline follows the
official recipe of [CLIP-not-BoW-unimodally](https://github.com/kdariina/CLIP-not-BoW-unimodally).

This research was supported by the MSIT(Ministry of Science, ICT), Korea, under the National Program for
Excellence in SW(2022-0-01077) supervised by the IITP(Institute of Information & communications Technology
Planning & Evaluation) in 2026.

## License

MIT, see [LICENSE](LICENSE). The vendored OpenCLIP keeps its own license
(`benchmarks/src/open_clip/LICENSE_OPENCLIP`). The images belong to their respective datasets and are not
redistributed here.
