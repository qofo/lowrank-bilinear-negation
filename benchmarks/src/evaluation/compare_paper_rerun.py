"""Put every number printed in the paper next to its reproduced value.

Reads the run folder written by scripts/reproduce_paper.sh and the reference outputs shipped in
results/reference/ (the logs the paper's numbers were taken from), and writes one row per reported
quantity:

    item, quantity, paper (as printed), reference (full precision from the original log),
    reproduced, reproduced - reference, reproduced rounded as the paper prints it == paper

Beyond the printed numbers it compares the trained matrices themselves (Ours rank-32 W and the
LABCLIP W against checkpoints/): max |W_reproduced - W_reference| says whether training reproduced
bit for bit or only approximately, which decides how far the downstream Table 3 numbers can move.

Stages that have not been run yet are reported as missing, so the script can be run at any point.

Usage (repo root):
    python -m benchmarks.src.evaluation.compare_paper_rerun --rerun outputs/paper
Outputs:
    <rerun>/compare_with_paper.csv
"""
import argparse
import json
import os

import pandas as pd
import torch

REF = "results/reference"
CKPT = "checkpoints"
T3_REF = {  # (benchmark, method) -> reference zero_shot_transfer_results.json
    ("COCO", "Ours"): f"{REF}/table3/coco_ours", ("COCO", "LABCLIP"): f"{REF}/table3/coco_labclip",
    ("VOC2007", "Ours"): f"{REF}/table3/voc_ours", ("VOC2007", "LABCLIP"): f"{REF}/table3/voc_labclip",
    ("CheXpert", "Ours"): f"{REF}/table3/chexpert_ours", ("CheXpert", "LABCLIP"): f"{REF}/table3/chexpert_labclip",
}
T3_CSV = {"COCO": "COCO_val_mcq_llama3.1_rephrased.csv", "VOC2007": "VOC2007_mcq_llama3.1_rephrased.csv",
          "CheXpert": "chexpert_binary_mcq_control_valid_only.csv"}
T3_CKPT = {"Ours": "delta_warmstart_rank32.pt", "LABCLIP": "labclip_official_recipe_bilinear.pt"}
T3_PAPER = {  # Avg (= total accuracy, mixed items included), Pos, Neg as printed in Table 3
    ("COCO", "Cosine"): (39.30, 69.14, 6.84), ("COCO", "LABCLIP"): (47.02, 82.23, 6.42),
    ("COCO", "Ours"): (45.25, 61.52, 24.22),
    ("VOC2007", "Cosine"): (38.72, 82.55, 3.35), ("VOC2007", "LABCLIP"): (40.95, 84.75, 6.46),
    ("VOC2007", "Ours"): (51.88, 73.61, 22.40),
    ("CheXpert", "Cosine"): (55.61,), ("CheXpert", "LABCLIP"): (37.43,), ("CheXpert", "Ours"): (58.82,),
}
T1_PAPER = {"random": (0.52, 0.54), "openai": (0.66, 0.86), "negft": (0.65, 0.93)}  # (Image, Text)
F2_PAPER_RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def probe_macro(run_dir):
    """Table 1 cell: per-object validation accuracy at the final L2-normalised layer, averaged over
    objects (macro), for the image and the text probe."""
    out = {}
    for modality, layer in (("vision", "+Final L2Norm"), ("text", "Final (L2 Normed)")):
        path = f"{run_dir}/beaf_{modality}_per_object_layerwise.csv"
        if not os.path.exists(path):
            out[modality] = out[f"{modality}_n"] = None
            continue
        df = pd.read_csv(path)
        sub = df[df.layer_name == layer]
        assert len(sub), (path, layer)
        out[modality] = sub.val_acc_pct.mean() / 100
        out[f"{modality}_n"] = sub.object_name.nunique()
    return out


def w_of(path):
    return torch.load(path, map_location="cpu")["state_dict"]["W"].double() if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerun", default="outputs/paper")
    args = ap.parse_args()
    R = args.rerun
    rows = []

    def add(item, quantity, paper, ref, rerun, digits):
        rows.append(dict(item=item, quantity=quantity, paper=paper, reference=ref, reproduced=rerun,
                         reproduced_minus_reference=None if rerun is None or ref is None else rerun - ref,
                         reproduced_prints_as_paper=None if rerun is None or paper is None
                         else round(rerun, digits) == round(paper, digits)))

    # Table 1: image probes on 76 objects (object-removed pairs), text probes on 69 objects.
    for tag in ("random", "openai", "negft"):
        o, r = probe_macro(f"{REF}/table1/{tag}"), probe_macro(f"{R}/table1_{tag}")
        for i, modality in enumerate(("vision", "text")):
            add("Table 1", f"{tag} {'image' if modality == 'vision' else 'text'} (macro over objects)",
                T1_PAPER[tag][i], o[modality], r[modality], 2)
            add("Table 1", f"{tag} {modality} n objects", None, o[f"{modality}_n"], r[f"{modality}_n"], 0)

    # Table 2 (the cosine row is also the 1.35% quoted in Sec 2.2)
    o, r = jload(f"{REF}/table2/per_object_intervention_summary.json"), \
        jload(f"{R}/table2/per_object_intervention_summary.json")
    for cond, label, p_cos, p_acc in (("1_Baseline_Cosine", "cosine", 0.219, 1.35),
                                      ("2_Closed_Form_Rotation", "rotation", 1.000, 3.25),
                                      ("7_Control_Random_Rotation", "random rotation", 0.002, 0.12)):
        acc_key = "acc_joint_mean_pct" if cond == "1_Baseline_Cosine" else "oof_acc_joint_mean_pct"
        add("Table 2", f"{label} cos(d_I, R d_T)", p_cos, o[cond]["direction_alignment_mean"],
            r[cond]["direction_alignment_mean"] if r else None, 3)
        add("Table 2" if cond != "1_Baseline_Cosine" else "Table 2 / Sec 2.2", f"{label} 2x2 acc %", p_acc,
            o[cond][acc_key], r[cond][acc_key] if r else None, 2)
    add("Table 2", "n objects", None, o["provenance"]["n_concepts_analyzed"],
        r["provenance"]["n_concepts_analyzed"] if r else None, 0)
    add("Table 2", "n image pairs", None, o["provenance"]["n_vision_pairs"],
        r["provenance"]["n_vision_pairs"] if r else None, 0)

    # Figure 2
    o, r = jload(f"{REF}/figure2/single_w_summary.json"), jload(f"{R}/figure2/single_w_summary.json")
    fo = {f["family"]: f for f in o["families"]}
    fr = {f["family"]: f for f in r["families"]} if r else {}
    add("Figure 2", "n pairs (Sec 2.1)", 2480, o["n_pairs"], r["n_pairs"] if r else None, 0)
    add("Figure 2", "n objects (Sec 2.1)", 42, o["n_concepts"], r["n_concepts"] if r else None, 0)
    add("Figure 2", "cosine (identity W) 2x2 acc % on these 42 objects", None,
        fo["identity"]["oof_pooled_acc_pct"], fr["identity"]["oof_pooled_acc_pct"] if r else None, 2)
    for rank in F2_PAPER_RANKS:
        fam = f"lowrank_{rank}"
        for key, label in (("oof_pooled_acc_pct", "unseen objects"), ("in_sample_acc_pct", "training objects")):
            add("Figure 2", f"r={rank} {label} %", round(fo[fam][key], 2), fo[fam][key],
                fr[fam][key] if r else None, 2)
    o = [x for x in jload(f"{REF}/figure2/labclip_rank_strategy_report.json")["conditions"]["A_posthoc_svd"]
         if x["rank"] == 512][0]["delta_s"]
    r = jload(f"{R}/figure2_labclip_star/delta_s_report.json")
    add("Figure 2", "LABCLIP star %", 10.04, o, r["labclip"] if r else None, 2)

    # The trained matrices
    for name, ref_path, new_path in (
            ("Ours rank-32 W", f"{CKPT}/delta_warmstart_rank32.pt", f"{R}/ours_rank32/delta_warmstart_rank32.pt"),
            ("LABCLIP W", f"{CKPT}/labclip_official_recipe_bilinear.pt",
             f"{R}/labclip/labclip_official_recipe_bilinear.pt")):
        wo, wr = w_of(ref_path), w_of(new_path)
        add("Trained W", f"{name} ||W||_F", None, wo.norm().item(), wr.norm().item() if wr is not None else None, 2)
        add("Trained W", f"{name} max|W_reproduced - W_reference|", None, 0.0,
            (wr - wo).abs().max().item() if wr is not None else None, 6)
    o, r = jload(f"{REF}/labclip_train/train_report.json"), jload(f"{R}/labclip/train_report.json")
    for k in ("n_pairs", "final_loss", "W_minus_I_frobenius"):
        add("LABCLIP training", k, None, o[k], r[k] if r else None, 4)

    # Table 3
    r = jload(f"{R}/table3/mcq_by_ckpt.json")
    metrics = ("total_accuracy", "positive_accuracy", "negative_accuracy")
    for bench in ("COCO", "VOC2007", "CheXpert"):
        for method in ("Cosine", "LABCLIP", "Ours"):
            src = T3_REF[(bench, "Ours")] if method == "Cosine" else T3_REF[(bench, method)]
            o = jload(f"{src}/zero_shot_transfer_results.json")
            o = o["Baseline Cosine"] if method == "Cosine" else o["Pretrained_bilinear"]
            rr = None
            if r:
                block = r[T3_CSV[bench]]
                rr = block["cosine"] if method == "Cosine" else block[T3_CKPT[method]]
            for metric, paper in zip(metrics, T3_PAPER[(bench, method)]):
                add("Table 3", f"{bench} {method} {metric.split('_')[0]}", paper, o[metric],
                    rr[metric] if rr else None, 2)

    # Figure 1
    o, r = jload(f"{REF}/figure1/fig1_example_pair.json"), jload(f"{R}/figures/fig1_example_pair.json")
    for k in ("Spp", "Spm", "Smp", "Smm", "delta"):
        add("Figure 1", k, round(o[k], 4 if k != "delta" else 3), o[k], r[k] if r else None,
            4 if k != "delta" else 3)

    df = pd.DataFrame(rows)
    out = f"{R}/compare_with_paper.csv"
    os.makedirs(R, exist_ok=True)
    df.to_csv(out, index=False)
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:.6g}"):
        print(df.to_string(index=False))
    done = df.reproduced.notna()
    printed = df[done & df.paper.notna()]
    print(f"\n{done.sum()}/{len(df)} quantities reproduced; "
          f"{int((printed.reproduced_prints_as_paper == True).sum())}/{len(printed)} print as the paper does")
    print("saved:", out)


if __name__ == "__main__":
    main()
