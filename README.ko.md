# 객체 단위 통제로 본 CLIP 부정 이해의 병목 분석과 저계수 이중선형 변환

[이준서](mailto:junseolee@ajou.ac.kr), [조현석](mailto:hyunsouk@ajou.ac.kr)\*
아주대학교 소프트웨어학과

**제7회 한국인공지능학술대회 (Korea AI 2026)**

[![Korea AI 2026](https://img.shields.io/badge/Korea%20AI-2026-blue.svg)](https://koreaai.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)
![PyTorch 2.8](https://img.shields.io/badge/PyTorch-2.8-ee4c2c.svg)

[English](README.md)

제7회 한국인공지능학술대회에 발표한 논문의 공식 코드입니다.

<p align="center"><img src="assets/fig1_minimal_pair.png" width="70%" alt="그림 1"></p>

CLIP이 부정이 들어간 질의를 잘 처리하지 못한다는 사실은 여러 연구에서 보고되었습니다. 하지만 모델의 어느 부분이 문제인지는 분명하지 않았습니다. 저희는 BEAF의 객체 제거 이미지로 이 문제를 살펴보았습니다. 같은 사진에서 서로 다른 객체를 지운 두 장을 짝짓고, 단어는 같지만 부정하는 객체가 다른 두 캡션("no cup, but a pizza" / "no pizza, but a cup")을 붙였습니다. 맞는 이미지–캡션 두 쌍의 점수가 틀린 두 쌍보다 모두 높을 때만 정답으로 봅니다. 점수를 무작위로 매겨도 16.67%는 맞히는데, 코사인 유사도는 1.35%에 그칩니다.

그래서 실패의 원인이 될 수 있는 세 곳을 차례로 확인했습니다. 먼저 선형 프로브는 이미지 임베딩에서 객체의 존재를, 텍스트 임베딩에서 부정을 무작위 초기화 인코더보다 분명히 잘 읽어냅니다. 필요한 정보는 이미 임베딩에 들어 있다는 뜻입니다. 다음으로 두 모달리티의 프로브 방향을 회전으로 완전히 맞춰도 정확도는 3.25%에 머무릅니다. 정렬만의 문제도 아닙니다. 남는 것은 유사도 함수입니다. 코사인 유사도 안의 항등 행렬을 모든 객체가 함께 쓰는 저계수 이중선형 변환 `S = (U e_I)^T (V e_T)`로 바꾸면, 학습에 쓰지 않은 객체에서도 무작위보다 높은 정확도가 나옵니다. 인코더를 그대로 둔 채로 NegBench MCQ의 부정 문항 정확도도 COCO에서 6.84%에서 24.22%로, VOC2007에서 3.35%에서 22.40%로 오릅니다.

## 소식

- **2026.10.01** 제7회 한국인공지능학술대회에서 발표했습니다(온라인 구두 발표).
- **2026.09** 코드를 공개했습니다.

## 결과

**표 1.** 객체별 선형 프로빙 정확도(객체 평균), CLIP ViT-B/32.

| 인코더 | 이미지 | 텍스트 |
|:--|:--:|:--:|
| 무작위 초기화 | 0.52 | 0.54 |
| OpenCLIP | 0.66 | 0.86 |
| OpenCLIP (ft) [NegBench] | 0.65 | 0.93 |

**표 2.** 이미지·텍스트 프로브 법선 사이의 코사인과 2×2 판정 정확도.

| 유사도 | cos(d_I, R d_T) | ΔS > 0 (%) |
|:--|:--:|:--:|
| 코사인(기준선) | 0.219 | 1.35 |
| 직교 회전 R_o | 1.000 | 3.25 |
| 무작위 회전 | 0.002 | 0.12 |

**그림 2.** 계수에 따른 2×2 판정 정확도. 실선은 학습에서 뺀 객체, 점선은 학습 객체, 별표는 학습 객체에서 잰 LABCLIP입니다.

<p align="center"><img src="assets/fig2_rank_curve.png" width="55%" alt="그림 2"></p>

**표 3.** 외부 벤치마크 정확도(%). 두 변환 모두 BEAF 쌍으로만 학습했습니다. Ours는 r = 32이며, Avg에는 혼합(hybrid) 문항도 들어갑니다.

| 유사도 | COCO Avg | COCO Pos | COCO Neg | VOC2007 Avg | VOC2007 Pos | VOC2007 Neg | CheXpert |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 무작위 | 25.00 | 25.00 | 25.00 | 25.00 | 25.00 | 25.00 | 50.00 |
| 코사인 | 39.30 | 69.14 | 6.84 | 38.72 | 82.55 | 3.35 | 55.61 |
| LABCLIP | **47.02** | **82.23** | 6.42 | 40.95 | **84.75** | 6.46 | 37.43 |
| **Ours** | 45.25 | 61.52 | **24.22** | **51.88** | 73.61 | **22.40** | **58.82** |

## 설치

```bash
git clone https://github.com/qofo/lowrank-bilinear-negation.git
cd lowrank-bilinear-negation
conda create -n lbn python=3.9 -y
conda activate lbn
pip install -r requirements.txt
```

NegBench가 수정한 OpenCLIP을 `benchmarks/src/open_clip`에 넣어 두었으며, pip로 따로 설치하지 않습니다. 아래 스크립트들은 `PYTHONPATH`를 알아서 설정합니다. 모듈을 직접 실행하실 때는 저장소 루트에서 먼저 다음을 실행해 주십시오.

```bash
export PYTHONPATH="$PWD:$PWD/benchmarks:$PWD/benchmarks/src"
```

## 데이터

CSV는 저장소에 들어 있습니다(`benchmarks/data/images/`). 이미지는 원래 출처에서 받으셔야 합니다.

- [BEAF v1](https://drive.google.com/file/d/1Xx7j8Hz8QX3Fl_hpSBet6r15njhwCgeR/view) (`beaf_dataset_ver1.zip`, BEAF 저자 배포본): 표 3을 뺀 모든 실험에 필요합니다.
- [COCO 2017 val 이미지](http://images.cocodataset.org/zips/val2017.zip): 표 3
- [PASCAL VOC2007 test](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar): 표 3
- CheXpert-v1.0-small 검증 분할: NegBench가 쓴 [Kaggle 미러](https://www.kaggle.com/datasets/ashery/chexpert)에서 받을 수 있으며 표 3에 쓰입니다. CheXpert 연구 이용 약관을 따라야 합니다.

받은 파일을 `prepare_data.sh`에 넘기면 CSV가 기대하는 위치에 풀고, 각 이미지를 `benchmarks/data/checksums/`의 SHA-256 목록과 대조합니다. 가지고 계신 파일만 넘기셔도 됩니다.

```bash
bash scripts/prepare_data.sh --beaf beaf_dataset_ver1.zip --coco_val2017 val2017.zip \
    --voc2007 VOCtest_06-Nov-2007.tar --chexpert /path/to/CheXpert-v1.0-small
```

**체크포인트.** OpenAI ViT-B/32 가중치는 OpenCLIP이 알아서 내려받습니다. 표 1의 미세조정 모델은 [NegBench 모델 폴더](https://drive.google.com/drive/folders/1kSEq0mkV1t1T8GuOAM65iz_iAA7e5gxB)에서 `CLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01`을 받아 `benchmarks/models/CLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01/checkpoint.pt`에 두시면 됩니다(sha256 `23a68e31da76f95670c4167cb2980e52fbd1eba77c8932cc36ade75efadfeb70`). 표 3에 쓴 두 행렬은 `checkpoints/`에 있습니다. Ours는 `delta_warmstart_rank32.pt`, LABCLIP은 `labclip_official_recipe_bilinear.pt`입니다.

## 논문 재현

```bash
bash scripts/reproduce_paper.sh all
```

단계별로 실행하실 수도 있습니다(`table1_openai`, `table2`, `figure2`, `ours`, `labclip`, `star`, `table3` 등). 전체 목록은 스크립트 맨 위에 있습니다. 결과는 `outputs/paper/`에 저장됩니다. 마지막 `compare` 단계는 `compare_with_paper.csv`를 만드는데, 논문에 인쇄된 각 수치를 원본 로그(`results/reference/`) 및 이번 실행 값과 나란히 보여 줍니다.

RTX 4070 한 장에서 전체 실행에 약 90분이 걸리며, 대부분은 표 1의 세 프로브입니다. 새 환경에서 다시 돌려 보았을 때 논문에 인쇄된 61개 수치가 모두 같게 나왔고, 학습한 두 행렬도 비트 단위로 일치했습니다. 표 1의 무작위 초기화 행만 값이 조금 다릅니다(0.5246 / 0.5441 대 0.5230 / 0.5354). 원래 실행에서 무작위 인코더에 시드를 주지 않았기 때문입니다. 반올림하면 논문과 같은 0.52 / 0.54이며, 지금 스크립트는 시드를 고정해 둡니다.

실험마다 쓰는 쌍이 조금씩 다르니 참고해 주십시오.

| 논문 항목 | 단계 | 사용한 쌍 |
|:--|:--|:--|
| 표 1 | `table1_*` | 이미지 프로브: 원본과 객체 제거본 쌍, 76개 객체(쌍 단위 GroupKFold). 텍스트 프로브: 단어 교환 캡션, 69개 객체(StratifiedKFold). |
| 2.2절(1.35%), 표 2 | `table2` | 원본과 객체 제거본 쌍, 33개 객체 1,357쌍 |
| 그림 2, 표 3(Ours) | `figure2`, `ours` | 한 사진에서 O와 다른 객체를 각각 지운 쌍, 42개 객체 2,480쌍 |
| LABCLIP | `labclip`, `star` | 같은 종류의 쌍 전체, 77개 객체 2,771쌍 |

학습 없이 공개한 행렬로 표 3만 채점하시려면 다음을 실행하십시오.

```bash
python -m benchmarks.src.evaluation.eval_w_ckpts_mcq \
    --ckpts checkpoints/delta_warmstart_rank32.pt checkpoints/labclip_official_recipe_bilinear.pt \
    --targets benchmarks/data/images/COCO_val_mcq_llama3.1_rephrased.csv \
              benchmarks/data/images/VOC2007_mcq_llama3.1_rephrased.csv \
              benchmarks/data/images/chexpert_binary_mcq_control_valid_only.csv \
    --output outputs/table3_released.json
```

테스트는 `python -m pytest benchmarks/tests`로 실행합니다. 이미지 없이 1분 정도면 끝납니다.

## 저장소 구조

```
benchmarks/src/open_clip/        NegBench가 수정한 OpenCLIP
benchmarks/src/analysis/         특징 추출과 공용 도구
benchmarks/src/evaluation/       실험 코드
benchmarks/src/data_generation/  BEAF에서 단어 교환 캡션 쌍을 만드는 코드
benchmarks/data/                 CSV와 이미지 체크섬
checkpoints/                     Ours(rank 32)와 LABCLIP의 학습된 W
results/reference/               논문 수치를 가져온 원본 결과
scripts/                         prepare_data.sh, reproduce_paper.sh
```

## 인용

```bibtex
@inproceedings{lee2026negationbottleneck,
  title     = {객체 단위 통제로 본 {CLIP} 부정 이해의 병목 분석과 저계수 이중선형 변환},
  author    = {이준서 and 조현석},
  booktitle = {제7회 한국인공지능학술대회},
  year      = {2026}
}
```

## 감사의 글

이 코드는 [NegBench](https://github.com/m1k2zoo/negbench)(코드 기반, MCQ 벤치마크, 미세조정 인코더)와 [OpenCLIP](https://github.com/mlfoundations/open_clip)을 바탕으로 만들었습니다. 이미지 쌍은 [BEAF](https://github.com/postech-ami/BEAF)에서 가져왔고, LABCLIP 기준선은 [CLIP-not-BoW-unimodally](https://github.com/kdariina/CLIP-not-BoW-unimodally)의 공식 코드를 따랐습니다.

본 연구는 2026년 과학기술정보통신부 및 정보통신기획평가원의 SW중심대학사업의 연구결과로 수행되었음(2022-0-01077)

## 라이선스

MIT 라이선스입니다([LICENSE](LICENSE)). 함께 들어 있는 OpenCLIP 코드는 원래 라이선스(`benchmarks/src/open_clip/LICENSE_OPENCLIP`)를 따릅니다. 이미지는 각 데이터셋에 속하며, 이 저장소에서는 배포하지 않습니다.
