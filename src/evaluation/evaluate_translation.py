"""
Evaluation script for English-to-Sindhi translation on held-out test set.
Computes BLEU, chrF++, Script Validity, and Latency metrics.
Outputs a structured evaluation report.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import sacrebleu

import config
from src.translation.translator import SindhiTranslator
from src.data.validate_dataset import DatasetValidator


def evaluate_translation_system(
    test_file: Path = config.PARALLEL_TEST_FILE,
    report_file: Path = config.REPORTS_DIR / "translation_evaluation_report.json",
    num_beams: int = 2,
) -> Dict[str, Any]:
    """Runs translation evaluation on held-out parallel test data."""
    print(f"[EvalTranslation] Loading test dataset from: {test_file}")
    df = pd.read_csv(test_file)

    translator = SindhiTranslator(lazy_init=False)

    sources: List[str] = []
    hypotheses: List[str] = []
    references: List[str] = []
    latencies: List[float] = []
    valid_script_count = 0

    print(f"[EvalTranslation] Translating {len(df)} test samples...")
    for idx, row in df.iterrows():
        en = str(row["english"]).strip()
        sd_ref = str(row["sindhi"]).strip()

        start_t = time.time()
        res = translator.translate_segment(en, num_beams=num_beams)
        lat = (time.time() - start_t) * 1000.0

        sd_hyp = res["sindhi"]

        sources.append(en)
        references.append(sd_ref)
        hypotheses.append(sd_hyp)
        latencies.append(lat)

        if DatasetValidator.contains_arabic_script(sd_hyp):
            valid_script_count += 1

        print(f"[{idx+1}/{len(df)}] EN: {en}")
        print(f"         HYP: {sd_hyp}")
        print(f"         REF: {sd_ref}")
        print(f"         Latency: {lat:.1f}ms")

    # Compute BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    # Compute chrF++
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)

    script_validity_rate = (valid_script_count / len(df)) * 100 if len(df) else 0.0
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    report = {
        "test_dataset": str(test_file),
        "total_test_samples": len(df),
        "model_used": config.DEFAULT_TRANSLATION_MODEL,
        "num_beams": num_beams,
        "bleu_score": round(float(bleu.score), 2),
        "chrf_score": round(float(chrf.score), 2),
        "script_validity_pct": round(script_validity_rate, 2),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "samples": [
            {"english": s, "reference": r, "hypothesis": h, "latency_ms": round(l, 2)}
            for s, r, h, l in zip(sources, references, hypotheses, latencies)
        ],
    }

    print("\n--- Translation Evaluation Results ---")
    print(f"BLEU: {report['bleu_score']}")
    print(f"chrF++: {report['chrf_score']}")
    print(f"Script Validity: {report['script_validity_pct']}%")
    print(f"Average Latency: {report['avg_latency_ms']:.1f} ms")

    os.makedirs(report_file.parent, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[EvalTranslation] Saved evaluation report -> {report_file}")

    return report


if __name__ == "__main__":
    evaluate_translation_system()
