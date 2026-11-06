"""
End-to-end checks of the paper pipeline that need no images, GPU or downloaded weights.

* The reference logs in results/reference/ reproduce every number printed in the paper,
  through the same comparison `reproduce_paper.sh compare` runs.
* The hand-written ViT residual walk behind the Table 1 image probes returns what the
  model's own `encode_image` returns.
* Opt-in (`-m pipeline`, a few minutes on CPU): every stage of scripts/reproduce_paper.sh
  runs as `python -m ...` on a tiny synthetic BEAF/MCQ set with a randomly initialised
  ViT-B/32, and the final `compare` stage finds every quantity it reads.
"""

import json
import os
import shutil
import subprocess
import sys
import zlib

import numpy as np
import pandas as pd
import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REF = os.path.join(REPO, "results", "reference")
CSV_DIR = os.path.join(REPO, "benchmarks", "data", "images")
MCQ_CSVS = ("COCO_val_mcq_llama3.1_rephrased.csv", "VOC2007_mcq_llama3.1_rephrased.csv",
            "chexpert_binary_mcq_control_valid_only.csv")


def _env():
    roots = [REPO, os.path.join(REPO, "benchmarks"), os.path.join(REPO, "benchmarks", "src")]
    if os.environ.get("PYTHONPATH"):
        roots.append(os.environ["PYTHONPATH"])
    return dict(os.environ, PYTHONPATH=os.pathsep.join(roots), CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")


def _run_compare(rerun_dir):
    proc = subprocess.run([sys.executable, "-m", "benchmarks.src.evaluation.compare_paper_rerun",
                           "--rerun", rerun_dir], cwd=REPO, env=_env(), capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    return pd.read_csv(os.path.join(rerun_dir, "compare_with_paper.csv"))


class TestReferenceLogsPrintAsPaper:
    """Laid out as a rerun, the shipped reference logs must print as every paper number."""

    @staticmethod
    def _reference_as_rerun(dst):
        def cp(src, rel):
            os.makedirs(os.path.dirname(os.path.join(dst, rel)), exist_ok=True)
            shutil.copy(src, os.path.join(dst, rel))

        for tag in ("random", "openai", "negft"):
            for modality in ("vision", "text"):
                name = f"beaf_{modality}_per_object_layerwise.csv"
                cp(f"{REF}/table1/{tag}/{name}", f"table1_{tag}/{name}")
        cp(f"{REF}/table2/per_object_intervention_summary.json", "table2/per_object_intervention_summary.json")
        cp(f"{REF}/figure2/single_w_summary.json", "figure2/single_w_summary.json")
        with open(f"{REF}/figure2/labclip_rank_strategy_report.json") as f:
            svd = json.load(f)["conditions"]["A_posthoc_svd"]
        os.makedirs(os.path.join(dst, "figure2_labclip_star"))
        with open(os.path.join(dst, "figure2_labclip_star", "delta_s_report.json"), "w") as f:
            json.dump({"labclip": [x for x in svd if x["rank"] == 512][0]["delta_s"]}, f)
        cp(f"{REPO}/checkpoints/delta_warmstart_rank32.pt", "ours_rank32/delta_warmstart_rank32.pt")
        cp(f"{REPO}/checkpoints/labclip_official_recipe_bilinear.pt", "labclip/labclip_official_recipe_bilinear.pt")
        cp(f"{REF}/labclip_train/train_report.json", "labclip/train_report.json")
        mcq = {}
        for bench, csv in zip(("coco", "voc", "chexpert"), MCQ_CSVS):
            with open(f"{REF}/table3/{bench}_ours/zero_shot_transfer_results.json") as f:
                ours = json.load(f)
            with open(f"{REF}/table3/{bench}_labclip/zero_shot_transfer_results.json") as f:
                labclip = json.load(f)
            mcq[csv] = {"cosine": ours["Baseline Cosine"],
                        "delta_warmstart_rank32.pt": ours["Pretrained_bilinear"],
                        "labclip_official_recipe_bilinear.pt": labclip["Pretrained_bilinear"]}
        os.makedirs(os.path.join(dst, "table3"))
        with open(os.path.join(dst, "table3", "mcq_by_ckpt.json"), "w") as f:
            json.dump(mcq, f)
        cp(f"{REF}/figure1/fig1_example_pair.json", "figures/fig1_example_pair.json")

    def test_every_printed_number_matches(self, tmp_path):
        rerun = str(tmp_path / "rerun")
        self._reference_as_rerun(rerun)
        df = _run_compare(rerun)
        assert df.reproduced.notna().all(), df[df.reproduced.isna()].to_string()
        assert float(df.reproduced_minus_reference.abs().max()) == 0.0
        printed = df[df.paper.notna()]
        assert len(printed) == 61
        mismatched = printed[printed.reproduced_prints_as_paper != True]  # noqa: E712 -- column holds objects
        assert mismatched.empty, mismatched.to_string()

    def test_released_ours_w_is_rank_32(self):
        # W = U^T V is formed in float32, so the trailing singular values are rounding noise
        # (~1e-6 against a leading value of ~226), not extra rank.
        W = torch.load(f"{REPO}/checkpoints/delta_warmstart_rank32.pt", map_location="cpu")["state_dict"]["W"]
        assert int(torch.linalg.matrix_rank(W, rtol=1e-5)) == 32


class TestManualVisionWalk:
    """The Table 1 image probes read a hand-written residual walk, not `encode_image`."""

    def test_final_embedding_matches_encode_image_and_ignores_batch(self, tmp_path):
        import torch.nn.functional as F
        from PIL import Image

        import open_clip
        from benchmarks.src.analysis.beaf.vision_mechanisms import extract_vision_features_unified

        torch.manual_seed(0)
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=None)
        model.eval()
        rng = np.random.default_rng(0)
        paths = []
        for i in range(5):
            p = str(tmp_path / f"{i}.png")
            Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(p)
            paths.append(p)

        out = extract_vision_features_unified(model, preprocess, paths, device="cpu", batch_size=4)
        with torch.no_grad():
            batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths])
            ref = F.normalize(model.encode_image(batch), dim=-1).numpy()
        np.testing.assert_allclose(out["final_l2norm"], ref, atol=1e-5)
        assert len(out["layers"]) == 13 and out["pre_proj"].shape == (5, 768)

        # Tokens attend within an image, never across the batch.
        solo = extract_vision_features_unified(model, preprocess, paths[:1], device="cpu", batch_size=4)
        np.testing.assert_allclose(solo["final_l2norm"][0], out["final_l2norm"][0], atol=1e-5)
        for key in out["layers"]:
            np.testing.assert_allclose(solo["layers"][key][0], out["layers"][key][0], atol=1e-4)


# ── opt-in: every reproduce_paper.sh stage on synthetic data ───────────────────────────────────

OBJECTS = ["pizza", "dog", "car", "cup", "chair", "bottle"]
FIG1 = ("pizza", "diverse_lack_neg_first_02_p1273")  # the pair compute_fig1_example.py looks up
RANK_FAMILIES = ["identity", "random", "diagonal", "lowrank_1", "lowrank_2", "lowrank_4", "lowrank_8",
                 "lowrank_16", "lowrank_32", "full", "lowrank_64", "lowrank_128", "lowrank_256", "lowrank_512"]


def _pair_rows(df, objects, n, extra=()):
    """Rows (2i, 2i+1) form one counterfactual pair; positions of the first n pairs of each object."""
    starts = set(extra)
    for obj in objects:
        starts |= set([i for i in range(0, len(df), 2) if df.object_name.iat[i] == obj][:n])
    return [j for i in sorted(starts) for j in (i, i + 1)]


def _write_image(root, rel_path):
    from PIL import Image

    full = os.path.join(root, rel_path)
    if not os.path.exists(full):
        os.makedirs(os.path.dirname(full), exist_ok=True)
        rng = np.random.default_rng(zlib.crc32(rel_path.encode()))
        Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(full)


def _build_fixture(work):
    """CSV subsets under work/csv and random images at the relative paths those CSVs name."""
    csv = os.path.join(work, "csv")
    os.makedirs(csv)
    by_concept = pd.read_csv(f"{CSV_DIR}/beaf_counterfactual_ab_swap_by_concept.csv")
    diverse = pd.read_csv(f"{CSV_DIR}/beaf_counterfactual_ab_swap_diverse.csv")
    six_col = pd.read_csv(f"{CSV_DIR}/beaf_counterfactual_6col.csv")
    fig1 = by_concept.index[(by_concept.object_name == FIG1[0]) & (by_concept.source_template == FIG1[1])][0]
    rows = _pair_rows(by_concept, OBJECTS, 4, extra=[fig1])
    sub_bc = by_concept.iloc[rows]
    sub_dv = diverse.iloc[rows]  # row-aligned with the by-concept CSV
    sub_6c = six_col.iloc[_pair_rows(six_col, OBJECTS, 3)]
    sub_bc.to_csv(f"{csv}/ab_swap_by_concept.csv", index=False)
    sub_dv.to_csv(f"{csv}/ab_swap_diverse.csv", index=False)
    sub_6c.to_csv(f"{csv}/six_col.csv", index=False)
    with open(f"{csv}/objects.txt", "w") as f:
        f.write("\n".join(OBJECTS) + "\n")
    paths = set(sub_bc.image_path) | set(sub_dv.image_path) | set(sub_6c.image_path)
    for name in MCQ_CSVS:
        mcq = pd.read_csv(f"{CSV_DIR}/{name}").head(16)
        mcq.to_csv(f"{csv}/{name}", index=False)
        paths |= set(mcq.image_path)
    for p in sorted(paths):
        _write_image(work, p)


def _stages(work, out):
    csv = os.path.join(work, "csv")
    cache = os.path.join(out, "_feature_cache")
    ab = os.path.join(csv, "ab_swap_by_concept.csv")
    model = ["--model", "ViT-B-32", "--pretrained", "", "--seed", "42"]
    beaf = [*model, "--image_root", work]
    ours = os.path.join(out, "ours_rank32", "delta_warmstart_rank32.pt")
    labclip = os.path.join(out, "labclip", "labclip_official_recipe_bilinear.pt")
    ev = "benchmarks.src.evaluation."
    figs = os.path.join(out, "figures")
    return [
        ("table1_random", ["-m", ev + "eval_probe_failure_inspector", *beaf, "--min_pairs", "2",
                           "--min_samples", "3", "--vision_csv", f"{csv}/six_col.csv",
                           "--text_csv", f"{csv}/ab_swap_diverse.csv", "--output_dir", f"{out}/table1_random"]),
        ("table2", ["-m", ev + "eval_per_object_alignment_intervention", *beaf, "--rank", "4", "--oof",
                    "--min_pairs", "2", "--vision_csv", f"{csv}/six_col.csv",
                    "--text_csv", f"{csv}/ab_swap_diverse.csv", "--restrict_objects", f"{csv}/objects.txt",
                    "--use_cache", "--cache_dir", cache, "--output_dir", f"{out}/table2"]),
        ("figure2", ["-m", ev + "eval_single_w_generalization", *beaf, "--csv_path", ab, "--min_pairs", "2",
                     "--loss", "delta", "--warmstart", "--epochs", "3", "--families", *RANK_FAMILIES,
                     "--use_cache", "--cache_dir", cache, "--output_dir", f"{out}/figure2"]),
        ("ours", ["-m", ev + "train_and_save_narrow_rank32_w", *beaf, "--csv_path", ab, "--min_pairs", "2",
                  "--epochs", "3", "--cache_dir", cache, "--output_dir", f"{out}/ours_rank32"]),
        ("labclip", ["-m", ev + "train_labclip_official_recipe", *model, "--csv_path", ab, "--epochs", "2",
                     "--batch_size", "8", "--output_dir", f"{out}/labclip"]),
        ("star", ["-m", ev + "score_delta_s_for_w", *beaf, "--ckpts", labclip, ours, "--names", "labclip", "ours",
                  "--csv_path", ab, "--min_pairs", "2", "--cache_dir", cache,
                  "--output_dir", f"{out}/figure2_labclip_star"]),
        ("table3", ["-m", ev + "eval_w_ckpts_mcq", *model, "--ckpts", ours, labclip,
                    "--targets", *[f"{csv}/{n}" for n in MCQ_CSVS], "--cache_dir", f"{out}/_mcq_cache",
                    "--output", f"{out}/table3/mcq_by_ckpt.json"]),
        ("figure1", [f"{REPO}/paper_figures/compute_fig1_example.py", *beaf, "--csv_path", ab,
                     "--output", f"{figs}/fig1_example_pair.json"]),
        ("figure1_plot", [f"{REPO}/paper_figures/mk_fig1_camera_ready.py", f"{figs}/fig1_example_pair.json",
                          f"{figs}/fig1_minimal_pair.png"]),
        ("figure2_plot", [f"{REPO}/paper_figures/mk_fig2_rank_heldout.py", "--run", f"{out}/figure2",
                          "--labclip", f"{out}/figure2_labclip_star/delta_s_report.json",
                          "--out", f"{figs}/fig2_rank_curve.png", "--data_out", f"{figs}/fig2_rank_curve_data.csv"]),
    ]


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    work = str(tmp_path_factory.mktemp("synthetic_paper_run"))
    out = os.path.join(work, "out")
    _build_fixture(work)
    results = {}
    for name, args in _stages(work, out):
        proc = subprocess.run([sys.executable, *args], cwd=work, env=_env(), capture_output=True, text=True)
        results[name] = proc
        if proc.returncode:
            break  # later stages read this one's outputs
    return out, results


@pytest.mark.pipeline
class TestPipelineOnSyntheticData:
    @pytest.mark.parametrize("stage", [name for name, _ in _stages("w", "o")])
    def test_stage_runs(self, synthetic_run, stage):
        _, results = synthetic_run
        assert stage in results, "an earlier stage failed"
        proc = results[stage]
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]

    def test_compare_finds_every_quantity(self, synthetic_run):
        out, results = synthetic_run
        assert all(p.returncode == 0 for p in results.values())
        df = _run_compare(out)
        # Only the random-init row of Table 1 runs here; the other two need downloaded weights.
        needs_weights = df.quantity.str.startswith(("openai ", "negft "))
        assert df[~needs_weights].reproduced.notna().all(), df[df.reproduced.isna()].to_string()

    def test_ours_w_has_rank_32(self, synthetic_run):
        out, _ = synthetic_run
        W = torch.load(os.path.join(out, "ours_rank32", "delta_warmstart_rank32.pt"),
                       map_location="cpu")["state_dict"]["W"]
        assert W.shape == (512, 512)
        assert int(torch.linalg.matrix_rank(W, rtol=1e-5)) <= 32
