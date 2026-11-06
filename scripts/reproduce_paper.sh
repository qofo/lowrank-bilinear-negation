#!/usr/bin/env bash
# Reproduce every table and figure of the paper from the raw images and the pretrained encoders.
#
# Fixed inputs: the OpenAI CLIP ViT-B/32 weights (fetched by OpenCLIP), the NegBench CLIP_CC12M_NegFull
# checkpoint, the CSVs under benchmarks/data/images/ and the images they point to (see README, "Data").
# Everything else -- embeddings, probes, rotations, the two trained matrices, benchmark scores and
# figures -- is recomputed under $OUT. Nothing is read from results/reference/ except by the final
# `compare` stage, which puts each recomputed number next to the one printed in the paper.
#
# Usage (from the repository root), one stage or several per call:
#     bash scripts/reproduce_paper.sh all
#     bash scripts/reproduce_paper.sh table2 figure2 ours labclip table3 compare
#
# Stages and the paper item each one reproduces:
#     table1_openai table1_random table1_negft
#                    Table 1   per-object linear probing (image: object-removed pairs, text: word-swapped captions)
#     table2         Table 2   probe-normal alignment and closed-form rotation (33 objects); also Sec 2.2's 1.35%
#     figure2        Figure 2  shared low-rank bilinear score, r = 1..512, GroupKFold over objects (42 objects)
#     ours           Table 3   Ours: rank-32 W trained with the Delta(S) hinge on all 42 objects
#     labclip        Table 3   LABCLIP: identity-initialised full W, text side, InfoNCE (official recipe)
#     star           Figure 2  2x2 accuracy of the LABCLIP W on its training objects (the star)
#     table3         Table 3   COCO / VOC2007 MCQ and CheXpert for cosine, Ours and LABCLIP
#     figure1_plot figure2_plot
#                    Figure 1 scores and drawing, Figure 2 drawing
#     compare        every paper number next to its reproduced value -> $OUT/compare_with_paper.csv
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD:$PWD/benchmarks/src${PYTHONPATH:+:$PYTHONPATH}"
# The reference numbers were produced with 4 BLAS/torch threads; other counts can move the last digits.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export MPLBACKEND=Agg

PYTHON="${PYTHON:-python}"
OUT="${OUT:-outputs/paper}"
FC="$OUT/_feature_cache"
LOGS="$OUT/_logs"
D=benchmarks/data/images
ABSWAP=$D/beaf_counterfactual_ab_swap_by_concept.csv
NEGFT=benchmarks/models/CLIP_CC12M_NegFull_ViT-B-32_lr1e-8_clw0.99_mlw0.01/checkpoint.pt
OURS=$OUT/ours_rank32/delta_warmstart_rank32.pt
LABCLIP=$OUT/labclip/labclip_official_recipe_bilinear.pt
mkdir -p "$FC" "$LOGS"

ALL_STAGES="table1_openai table1_random table1_negft table2 figure2 ours labclip star table3 figure1_plot figure2_plot compare"

# Table 1 keeps every object with at least 2 image pairs (76 objects) and every object with at least
# 10 captions per polarity (69 objects), the thresholds of the run the paper reports.
table1() {
    "$PYTHON" -m benchmarks.src.evaluation.eval_probe_failure_inspector \
        --model ViT-B-32 --pretrained "$1" --seed 42 --min_pairs 2 --min_samples 10 \
        --output_dir "$OUT/$2"
}

run_stage() {
    case "$1" in
    table1_openai) table1 openai table1_openai ;;
    table1_random) table1 '' table1_random ;;
    table1_negft)  table1 "$NEGFT" table1_negft ;;
    table2)   "$PYTHON" -m benchmarks.src.evaluation.eval_per_object_alignment_intervention \
                  --model ViT-B-32 --pretrained openai --seed 42 --rank 32 --oof \
                  --restrict_objects concept_sets/paper33.txt \
                  --use_cache --cache_dir "$FC" --output_dir "$OUT/table2" ;;
    figure2)  "$PYTHON" -m benchmarks.src.evaluation.eval_single_w_generalization \
                  --model ViT-B-32 --pretrained openai --seed 42 --csv_path $ABSWAP --loss delta --warmstart \
                  --families identity random diagonal lowrank_1 lowrank_2 lowrank_4 lowrank_8 lowrank_16 \
                             lowrank_32 full lowrank_64 lowrank_128 lowrank_256 lowrank_512 \
                  --use_cache --cache_dir "$FC" --output_dir "$OUT/figure2" ;;
    ours)     "$PYTHON" -m benchmarks.src.evaluation.train_and_save_narrow_rank32_w \
                  --csv_path $ABSWAP --seed 42 --cache_dir "$FC" --output_dir "$OUT/ours_rank32" ;;
    labclip)  "$PYTHON" -m benchmarks.src.evaluation.train_labclip_official_recipe \
                  --csv_path $ABSWAP --seed 42 --output_dir "$OUT/labclip" ;;
    star)     "$PYTHON" -m benchmarks.src.evaluation.score_delta_s_for_w --ckpts $LABCLIP $OURS \
                  --names labclip ours --csv_path $ABSWAP --seed 42 --cache_dir "$FC" \
                  --output_dir "$OUT/figure2_labclip_star" ;;
    table3)   "$PYTHON" -m benchmarks.src.evaluation.eval_w_ckpts_mcq --ckpts $OURS $LABCLIP \
                  --targets $D/COCO_val_mcq_llama3.1_rephrased.csv $D/VOC2007_mcq_llama3.1_rephrased.csv \
                            $D/chexpert_binary_mcq_control_valid_only.csv \
                  --seed 42 --cache_dir "$OUT/_mcq_cache" --output "$OUT/table3/mcq_by_ckpt.json" ;;
    figure1_plot)
              "$PYTHON" paper_figures/compute_fig1_example.py --output "$OUT/figures/fig1_example_pair.json" &&
              "$PYTHON" paper_figures/mk_fig1_camera_ready.py "$OUT/figures/fig1_example_pair.json" \
                  "$OUT/figures/fig1_minimal_pair.png" ;;
    figure2_plot)
              "$PYTHON" paper_figures/mk_fig2_rank_heldout.py --run "$OUT/figure2" \
                  --labclip "$OUT/figure2_labclip_star/delta_s_report.json" \
                  --out "$OUT/figures/fig2_rank_curve.png" --data_out "$OUT/figures/fig2_rank_curve_data.csv" ;;
    compare)  "$PYTHON" -m benchmarks.src.evaluation.compare_paper_rerun --rerun "$OUT" ;;
    *)        echo "unknown stage: $1 (stages: $ALL_STAGES)" >&2; return 2 ;;
    esac
}

stages=("$@")
[ "${#stages[@]}" -eq 0 ] && { echo "usage: bash scripts/reproduce_paper.sh all | <stage> ..." >&2; exit 2; }
[ "${stages[0]}" = "all" ] && read -r -a stages <<< "$ALL_STAGES"

for stage in "${stages[@]}"; do
    echo "[$(date '+%F %T')] start $stage"
    if run_stage "$stage" > "$LOGS/$stage.log" 2>&1; then
        echo "[$(date '+%F %T')] done  $stage"
    else
        echo "[$(date '+%F %T')] FAILED $stage (see $LOGS/$stage.log)" >&2
        exit 1
    fi
done
