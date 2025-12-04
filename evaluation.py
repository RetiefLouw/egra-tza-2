#!/usr/bin/env python3
"""Command-line entry point for running the simplified EGRA evaluation pipeline."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Import the local dp_align module provided in the context
try:
    import dp_align
except ImportError:
    raise ImportError("Could not import 'dp_align'. Ensure dp_align.py is in the script directory.")

from egra_eval.data.dataset_layout import DatasetLayoutError, resolve_dataset_paths
from egra_eval.data.linking import add_audio_keys, attach_hypotheses
from egra_eval.data.nemo_manifest import load_many_manifests
from egra_eval.data.passage_merge import attach_passage_texts
from egra_eval.data.textgrid_io import add_refs_from_textgrid
from egra_eval.eval.run_eval import evaluate
from egra_eval.report.summarize import (
    summary_for_pair,
    summary_per_speaker,
    summary_per_speaker_macro,
    summary_per_speaker_subcategory,
)

CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


def _append_a_to_consonant_letters(text: str) -> str:
    if text is None or pd.isna(text):
        return text
    parts = re.split(r"(\s+)", str(text))

    def transform_token(token: str) -> str:
        cleaned = re.sub(r"[^a-z]", "", token.lower())
        if not cleaned:
            return token
        first_char = cleaned[0]
        if first_char in CONSONANTS:
            if token.lower().endswith("a"):
                return token
            return f"{token}a"
        return token

    return "".join(
        transform_token(part) if idx % 2 == 0 else part
        for idx, part in enumerate(parts)
    )


def adjust_letter_canonical_text(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if "audio_type" not in df.columns or "canonical_text" not in df.columns:
        return df
    mask = df["audio_type"].astype(str).str.contains("letter", case=False, na=False)
    if not mask.any():
        return df
    logger.info("Adjusting canonical texts for %d letter row(s).", int(mask.sum()))
    out = df.copy()
    out.loc[mask, "canonical_text"] = out.loc[mask, "canonical_text"].apply(_append_a_to_consonant_letters)
    return out

# ---------------------------------------------------------------------------
# Advanced Metrics & Alignment Logic (dp_align Integration)
# ---------------------------------------------------------------------------

def get_csid_sequence(canonical_text: str, other_text: str) -> List[str]:
    """
    Uses dp_align to generate the sequence of 'c', 's', 'i', 'd' tags.
    
    Args:
        canonical_text: The "Ground Truth" / Reference list for the alignment.
        other_text: The "Test" / Hypothesis list for the alignment.
        
    Returns:
        A list of strings ['c', 's', 'i', 'c', 'd', ...] representing the alignment path.
    """
    # Handle NaNs or non-strings
    if pd.isna(canonical_text): canonical_text = ""
    if pd.isna(other_text): other_text = ""
    
    # dp_align works on lists of tokens (words), so we split.
    can_list = str(canonical_text).split()
    other_list = str(other_text).split()
    
    # dp_align(ref_list, test_list) -> We treat Canonical as Ref, Other as Test.
    # Returns (errors, alignment). Alignment is list of tuples: (test_token, ref_token, tag)
    _, alignment = dp_align.dp_align(can_list, other_list, output_align=True)
    
    # Extract the tag (3rd element) from the alignment tuple
    # alignment structure: [(test_token, ref_token, 'c'), ...]
    return [item[2] for item in alignment]


def compute_fine_grained_metrics(row) -> pd.Series:
    """
    Computes MER, P/R/F1 for substitutions, insertions, deletions, and all mistakes.
    Uses Alignment of Alignments (Meta-Alignment).
    """
    can = row.get('canonical_text', '')
    ref = row.get('reference_text', '')
    hyp = row.get('hypothesis_text', '')

    # 1. Generate Sequence A (Reference vs Canonical) and B (Hypothesis vs Canonical)
    # These are lists of 'c', 's', 'i', 'd'
    seq_a = get_csid_sequence(can, ref)
    seq_b = get_csid_sequence(can, hyp)
    
    # 2. Meta-Alignment: Align Sequence B (Pred) to Sequence A (Truth)
    # We treat seq_a as the "Ref List" and seq_b as the "Test List" in dp_align terms
    mer_errors, mer_alignment = dp_align.dp_align(seq_a, seq_b, output_align=True)
    
    # 3. Calculate Mistake Error Rate (MER)
    # MER is essentially the WER of the two mistake sequences
    mer_val = mer_errors.get_wer()

    # 4. Prepare data for Scikit-Learn P/R/F1
    # mer_alignment structure: [(seq_b_token, seq_a_token, align_type), ...]
    # y_pred comes from seq_b (Hypothesis mistake types)
    # y_true comes from seq_a (Reference mistake types)
    
    y_pred = [x[0] for x in mer_alignment]
    y_true = [x[1] for x in mer_alignment]
    
    # Note: dp_align uses '-' for gaps. 
    # e.g. If Reference had 's' and Hyp had nothing (deletion of mistake), y_true='s', y_pred='-'
    
    metrics_res = {}
    labels = ['s', 'i', 'd']
    
    # Calculate P/R/F1 for specific mistake types
    # We use `labels` to force sklearn to only look at s, i, d.
    # '-' or 'c' are treated as "not the class of interest"
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    
    for i, label in enumerate(labels):
        metrics_res[f'{label.upper()}_Precision'] = p[i]
        metrics_res[f'{label.upper()}_Recall'] = r[i]
        metrics_res[f'{label.upper()}_F1'] = f1[i]

    # 5. "All Mistakes" (Binary: Mistake vs Correct)
    # Convert s, i, d to 'x'. Convert 'c' and '-' to 'c' (or ignore).
    # We want to know: If it WAS a mistake (x), did we predict a mistake (x)?
    
    def binarize(token):
        if token in ['s', 'i', 'd']:
            return 'x'
        return 'c' # Treat 'c' and '-' as correct/neutral

    y_true_bin = [binarize(t) for t in y_true]
    y_pred_bin = [binarize(t) for t in y_pred]
    
    p_x, r_x, f1_x, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, labels=['x'], zero_division=0)
    
    metrics_res['Mistakes_Precision'] = p_x[0]
    metrics_res['Mistakes_Recall'] = r_x[0]
    metrics_res['Mistakes_F1'] = f1_x[0]
    metrics_res['MER'] = mer_val

    return pd.Series(metrics_res)


def calculate_advanced_metrics(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Enriches the dataframe with:
    1. MAE_EGRA_ACC: |ACC_can_ref - ACC_can_hyp| (Absolute Percentage Difference)
    2. Bias Baseline: MAE between actual ACC and the dataset average ACC.
    3. MER and Fine-grained P/R/F1 metrics using dp_align.
    """
    logger.info("Calculating advanced metrics (MAE, Bias, MER, P/R/F1) using dp_align...")
    
    # 1. MAE_EGRA_ACC
    # Ensure both ACC columns are on the same scale (0..1). Some codepaths return percentages (0..100).
    if "ACC_can_ref" in df.columns and "ACC_can_hyp" in df.columns:
        # Detect percent-scale (values > 1.0) and convert to 0..1
        ref_max = df["ACC_can_ref"].dropna().max() if not df["ACC_can_ref"].dropna().empty else 0
        hyp_max = df["ACC_can_hyp"].dropna().max() if not df["ACC_can_hyp"].dropna().empty else 0
        if ref_max > 1.0 or hyp_max > 1.0:
            logger.info("Detected ACC in 0-100 scale; converting ACC_can_ref/ACC_can_hyp to 0..1 scale for MAE calculation.")
            df["ACC_can_ref_norm_for_mae"] = df["ACC_can_ref"] / 100.0
            df["ACC_can_hyp_norm_for_mae"] = df["ACC_can_hyp"] / 100.0
        else:
            df["ACC_can_ref_norm_for_mae"] = df["ACC_can_ref"]
            df["ACC_can_hyp_norm_for_mae"] = df["ACC_can_hyp"]

        # Compute per-row MAE in the normalized (0..1) space
        df["MAE_EGRA_ACC"] = (df["ACC_can_ref_norm_for_mae"] - df["ACC_can_hyp_norm_for_mae"]).abs()
        # Optionally keep MAE in percent points for reporting by multiplying by 100 later in summary writer.
    else:
        df["MAE_EGRA_ACC"] = np.nan

    # 2. Bias Baseline
    # Predict the average ACC_can_ref for everyone
    if "ACC_can_ref" in df.columns:
        mean_acc = df["ACC_can_ref"].mean()
        # MAE between actual and mean
        df["Bias_Baseline_MAE"] = (df["ACC_can_ref"] - mean_acc).abs()
        logger.info(f"Bias Baseline (Mean Accuracy across dataset): {mean_acc:.4f}")
    else:
        df["Bias_Baseline_MAE"] = np.nan

    # 3. Fine-grained Metrics (MER, P, R, F1)
    req_cols = ["canonical_text", "reference_text", "hypothesis_text"]
    if all(c in df.columns for c in req_cols):
        # Filter to ensure we don't crash on empty rows, though get_csid handles it
        logger.info("Running alignment-of-alignments (MER/Fine-grained)...")
        fine_grained = df.apply(compute_fine_grained_metrics, axis=1)
        df = pd.concat([df, fine_grained], axis=1)
    
    return df

# ---------------------------------------------------------------------------
# Logging / CLI
# ---------------------------------------------------------------------------

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("egra_eval")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EGRA evaluation using canonical/ASR data.")
    p.add_argument("--dataset_root", required=True, help="Root folder containing 0_Audio/2_TextGrid and Student_* CSVs.")
    p.add_argument("--output_root", required=True, help="Directory where outputs will be written (detailed CSV + summary subfolders).")

    p.add_argument("--egra_csv", default=None)
    p.add_argument("--meta_csv", default=None)
    p.add_argument(
        "--passages_csv",
        required=True,
        help="CSV mapping passage numbers to canonical text (e.g. oral_passages.csv).",
    )
    p.add_argument("--nemo_manifest", action="append", default=None, help="Path(s) to NeMo JSON manifests with ASR hypotheses (can be supplied multiple times).")
    p.add_argument("--manifest_audio_key", default="audio_filepath")
    p.add_argument("--manifest_hyp_key", default="pred_text")
    p.add_argument("--manifest_can_key", default=None)
    p.add_argument("--match_on", choices=["stem", "name", "path"], default="stem", help="How to join manifest HYPs to EGRA rows.")

    p.add_argument("--out_csv", default=None)
    p.add_argument("--summary_can_ref_dir", default=None)
    p.add_argument("--summary_can_hyp_dir", default=None)
    p.add_argument("--summary_ref_hyp_dir", default=None)
    return p.parse_args()

# ---------------------------------------------------------------------------
# IO resolution / outputs
# ---------------------------------------------------------------------------

def resolve_inputs(args: argparse.Namespace, logger: logging.Logger) -> tuple[Path, Dict[str, Path]]:
    """Resolve dataset layout and output dirs; mutate args with defaults."""
    try:
        layout = resolve_dataset_paths(args.dataset_root)
    except DatasetLayoutError as exc:
        raise SystemExit(str(exc)) from exc

    # Fill inferred CSVs
    args.egra_csv = args.egra_csv or str(layout.canonical_csv)
    args.meta_csv = args.meta_csv or str(layout.metadata_csv)
    passages_path = Path(args.passages_csv)
    if not passages_path.exists():
        raise SystemExit(f"--passages_csv file not found: {passages_path}")

    # Resolve manifest(s)
    if not args.nemo_manifest:
        candidate = layout.root / "nemo_asr_output" / "transcriptions.jsonl"
        if candidate.exists():
            args.nemo_manifest = [str(candidate)]
        else:
            raise SystemExit(
                "No --nemo_manifest supplied and no default manifest found at "
                f"{candidate}. Run inference first or provide the path explicitly."
            )

    logger.info(
        "Dataset root resolved: audio=%s | textgrids=%s | canonical=%s | metadata=%s",
        layout.audio_root, layout.textgrid_root, layout.canonical_csv, layout.metadata_csv,
    )

    # Prepare outputs
    base = Path(args.output_root)
    base.mkdir(parents=True, exist_ok=True)

    args.out_csv = args.out_csv or str(base / "egra_eval_detailed.csv")
    args.summary_can_ref_dir = args.summary_can_ref_dir or str(base / "can_ref")
    args.summary_can_hyp_dir = args.summary_can_hyp_dir or str(base / "can_hyp")
    args.summary_ref_hyp_dir = args.summary_ref_hyp_dir or str(base / "ref_hyp")

    summary_dirs = {
        "can_ref": Path(args.summary_can_ref_dir),
        "can_hyp": Path(args.summary_can_hyp_dir),
        "ref_hyp": Path(args.summary_ref_hyp_dir),
    }
    return layout.textgrid_root, summary_dirs


def ensure_output_dirs(out_csv: str, summary_dirs: Dict[str, Path]) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    for d in summary_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data loading / writing
# ---------------------------------------------------------------------------

def load_manifest(args: argparse.Namespace, logger: logging.Logger) -> pd.DataFrame:
    if not args.nemo_manifest:
        return pd.DataFrame()
    logger.info("Loading %d NeMo manifest(s)...", len(args.nemo_manifest))
    df = load_many_manifests(
        args.nemo_manifest,
        audio_key=args.manifest_audio_key,
        hyp_key=args.manifest_hyp_key,
        can_key=args.manifest_can_key,
    )
    logger.info("Loaded manifest rows: %d", len(df))
    return df


def write_detailed_csv(df: pd.DataFrame, path: str, logger: logging.Logger) -> None:
    rename_map = {
        "ACC_can_ref": "ACC_can_ref (EGRA_ACC)",
        "C_can_ref": "C_can_ref (EGRA_COR)",
        "ACC_can_hyp": "ACC_can_hyp (ASR_EGRA_ACC)",
        "C_can_hyp": "C_can_hyp (ASR_EGRA_COR)",
        "EGRA_COR": "EGRA-COR",
        "EGRA_ACC": "EGRA-ACC",
        "ASR_EGRA_COR": "ASR-EGRA-COR",
        "ASR_EGRA_ACC": "ASR-EGRA-ACC",
        "MAE_EGRA_COR": "MAE_EGRA_COR",
        "ASR_WER": "ASR_WER",
        # New metrics
        "MAE_EGRA_ACC": "MAE_EGRA_ACC",
        "Bias_Baseline_MAE": "Bias_Baseline_MAE",
        "MER": "Mistake_Error_Rate (MER)",
    }
    df.rename(columns=rename_map).to_csv(path, index=False)
    logger.info("Wrote detailed results -> %s (rows: %d)", path, len(df))


def write_summary_csvs(df: pd.DataFrame, summary_dirs: Dict[str, Path], logger: logging.Logger) -> List[Path]:
    written: List[Path] = []

    def _write_one(pair: str, directory: Path) -> None:
        per_speaker = summary_per_speaker(df, pair)
        global_row = summary_for_pair(df, pair, by=None)
        global_row.insert(0, "learner_id", "__GLOBAL__")

        combined = (
            global_row
            if per_speaker.empty
            else pd.concat(
                [global_row[per_speaker.columns], per_speaker],
                ignore_index=True,
            )
        )
        per_speaker_path = directory / "egra_eval_summary_per_speaker_global.csv"
        combined.to_csv(per_speaker_path, index=False)
        logger.info("[%s] Wrote per-speaker summary -> %s", pair, per_speaker_path)
        written.append(per_speaker_path)

        macro_path = directory / "egra_eval_summary_per_speaker_macro.csv"
        summary_per_speaker_macro(df, pair).to_csv(macro_path, index=False)
        logger.info("[%s] Wrote per-speaker × macro-category summary -> %s", pair, macro_path)
        written.append(macro_path)

        subcat_path = directory / "egra_eval_summary_per_speaker_subcat.csv"
        summary_per_speaker_subcategory(df, pair).to_csv(subcat_path, index=False)
        logger.info("[%s] Wrote per-speaker × subcategory summary -> %s", pair, subcat_path)
        written.append(subcat_path)

    for pair, directory in summary_dirs.items():
        _write_one(pair, directory)

    return written


def write_text_summary(df: pd.DataFrame, out_csv: str, logger: logging.Logger) -> Path:
    summary_path = Path(out_csv).parent / "egra_eval_summary.txt"

    metrics: Dict[str, float] = {
        "EGRA-COR": float("nan"),
        "EGRA-ACC": float("nan"),
        "ASR-EGRA-COR": float("nan"),
        "ASR-EGRA-ACC": float("nan"),
        "MAE_EGRA_COR": float("nan"),
        "MAE_EGRA_ACC": float("nan"),
        "Bias_Baseline_MAE": float("nan"),
        "MER": float("nan"),
        "Mistakes_F1": float("nan"),
        "ASR_WER": float("nan"),
    }

    can_ref = summary_for_pair(df, "can_ref")
    if not can_ref.empty:
        metrics["EGRA-COR"] = can_ref["C_can_ref"].iloc[0]
        metrics["EGRA-ACC"] = can_ref["ACC_can_ref"].iloc[0]

    can_hyp = summary_for_pair(df, "can_hyp")
    if not can_hyp.empty:
        metrics["ASR-EGRA-COR"] = can_hyp["C_can_hyp"].iloc[0]
        metrics["ASR-EGRA-ACC"] = can_hyp["ACC_can_hyp"].iloc[0]

    # Averages for row-based metrics
    # Note: MAE_EGRA_ACC is calculated per row in 0..1 space; display as percentage.
    for m in ["MAE_EGRA_COR", "MAE_EGRA_ACC", "Bias_Baseline_MAE", "MER", "Mistakes_F1"]:
        if m in df.columns and not df[m].dropna().empty:
            val = float(df[m].dropna().mean())
            if m == "MAE_EGRA_ACC":
                val = val * 100.0  # present MAE_EGRA_ACC as percentage
            metrics[m] = val

    ref_hyp = summary_for_pair(df, "ref_hyp")
    if not ref_hyp.empty:
        metrics["ASR_WER"] = ref_hyp["WER_ref_hyp"].iloc[0]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Metric\tValue\n")
        for key in metrics.keys():
            value = metrics[key]
            if isinstance(value, float):
                if pd.isna(value):
                    formatted = "NaN"
                else:
                    # Use at most 2 decimal places for floats
                    formatted = f"{value:.2f}"
            else:
                formatted = str(value)
            f.write(f"{key}\t{formatted}\n")

    logger.info("Wrote text summary -> %s", summary_path)
    return summary_path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger = setup_logger()
    args = parse_args()

    textgrid_root, summary_dirs = resolve_inputs(args, logger)
    ensure_output_dirs(args.out_csv, summary_dirs)

    logger.info("Starting EGRA evaluation pipeline")
    logger.info(
        "EGRA CSV: %s | META CSV: %s | Passages CSV: %s | TextGrid root: %s | Manifest(s): %s",
        args.egra_csv, args.meta_csv, args.passages_csv, textgrid_root, args.nemo_manifest,
    )

    df_egra = pd.read_csv(args.egra_csv)
    df_meta = pd.read_csv(args.meta_csv)
    logger.info("Loaded EGRA rows: %d | META rows: %d", len(df_egra), len(df_meta))

    df_egra = adjust_letter_canonical_text(df_egra, logger)
    df_egra = add_audio_keys(df_egra, audio_col="audio_file")
    manifest_df = load_manifest(args, logger)
    df_egra = attach_hypotheses(df_egra, manifest_df, match_on=args.match_on)
    df_egra = add_refs_from_textgrid(
        df_egra,
        base_dir=str(textgrid_root),
        textgrid_col="textgrid",
        tier_name="child",
        logger=logger,
    )

    logger.info("Attaching passage texts from %s", args.passages_csv)
    df_egra = attach_passage_texts(df_egra, args.passages_csv, logger=logger)

    # Run Basic Evaluation (Jiwer based counts for standard WER/ACC)
    df_results = evaluate(df_egra, df_meta)
    
    # --- NEW: Run Advanced Metrics using dp_align ---
    df_results = calculate_advanced_metrics(df_results, logger)
    # ------------------------------------------------

    write_detailed_csv(df_results, args.out_csv, logger)
    summary_txt_path = write_text_summary(df_results, args.out_csv, logger)
    summary_paths = write_summary_csvs(df_results, summary_dirs, logger)

    print("Detailed results:", args.out_csv)
    print("Summary text:", summary_txt_path)
    print("Summary directories/files:")
    for path in summary_paths:
        print("  ", path)


if __name__ == "__main__":
    main()