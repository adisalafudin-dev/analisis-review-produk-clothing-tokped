"""
Master Pipeline Runner
======================
Runs the end-to-end review analysis pipeline in sequential order.

Usage:
    # Run full pipeline starting from data cleaning (using existing scraped data):
    python run_pipeline.py

    # Run full pipeline including web scraping:
    python run_pipeline.py --scrape
"""

import sys
import subprocess
import time

STEPS = [
    {
        "step": 1,
        "name": "Data Scraping",
        "script": "scraping.py",
        "description": "Scraping reviews from Tokopedia via GraphQL",
        "optional": True,
    },
    {
        "step": 2,
        "name": "Data Cleaning",
        "script": "data-cleaning.py",
        "description": "HTML unescape, deduplication, and template boilerplate removal",
        "optional": False,
    },
    {
        "step": 3,
        "name": "Data Labelling",
        "script": "data-labelling.py",
        "description": "Label reviews into 'baik' (rating 4-5) and 'buruk' (rating 1-3)",
        "optional": False,
    },
    {
        "step": 4,
        "name": "NLP Preprocessing",
        "script": "data-preprocessing.py",
        "description": "Case folding, cleaning, slang normalization, stopwords & Sastrawi stemming",
        "optional": False,
    },
    {
        "step": 5,
        "name": "Data Splitting",
        "script": "data-splitting.py",
        "description": "Stratified 80/20 train-test split preserving class ratio",
        "optional": False,
    },
    {
        "step": 6,
        "name": "Feature Extraction",
        "script": "data-extraction.py",
        "description": "Generate BoW, TF-IDF, Word2Vec, and Doc2Vec representations",
        "optional": False,
    },
    {
        "step": 7,
        "name": "XGBoost Modeling",
        "script": "modeling_xgboost.py",
        "description": "Train XGBoost with scale_pos_weight for all 4 feature representations",
        "optional": False,
    },
    {
        "step": 8,
        "name": "Model Evaluation",
        "script": "evaluasi_final.py",
        "description": "Classification report, confusion matrices, and 3-fold cross-validation",
        "optional": False,
    },
    {
        "step": 9,
        "name": "Word Cloud Visualizations",
        "script": "wordcloud_ecommerce.py",
        "description": "Generate comparative word clouds and top-15 keyword tables",
        "optional": False,
    },
]


def run_command(script_name):
    """Run a python script using the current python executable."""
    cmd = [sys.executable, script_name]
    res = subprocess.run(cmd)
    return res.returncode == 0


def main():
    include_scrape = "--scrape" in sys.argv

    print("=" * 70)
    print(" [PIPELINE] TOKOPEDIA CLOTHING REVIEW SENTIMENT ANALYSIS")
    print("=" * 70)
    print(f"Python interpreter: {sys.executable}")
    print(f"Include scraping stage: {'Yes' if include_scrape else 'No (using existing raw data)'}")
    print("=" * 70)

    start_time = time.time()

    for item in STEPS:
        if item["step"] == 1 and not include_scrape:
            print(f"\n[SKIP] Step 1: {item['name']} (use --scrape flag to run scraping)")
            continue

        print(f"\n>>> Running Step {item['step']}: {item['name']} ({item['script']})")
        print(f"    {item['description']}")
        print("-" * 70)

        step_start = time.time()
        success = run_command(item["script"])
        elapsed = time.time() - step_start

        if not success:
            print(f"\n[FAIL] Pipeline failed at Step {item['step']} ({item['script']})!")
            sys.exit(1)

        print(f"[OK] Step {item['step']} completed successfully in {elapsed:.1f}s.")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.1f}s!")
    print("=" * 70)
    print("Generated Artifacts:")
    print("  - reviews_tokopedia_clean.csv")
    print("  - dataset_labeled.csv & dictionary_output.pdf")
    print("  - dataset_preprocessed.csv")
    print("  - data_train.csv & data_test.csv")
    print("  - fitur_semua_metode.pkl (BoW, TF-IDF, Word2Vec, Doc2Vec)")
    print("  - hasil_model_xgboost.pkl")
    print("  - evaluasi_confusion_matrix.png & evaluasi_perbandingan_metode.png")
    print("  - wordcloud_baik_buruk.png")


if __name__ == "__main__":
    main()
