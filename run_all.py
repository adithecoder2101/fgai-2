"""
Master Pipeline Runner for ICT4403 Assignment 3.
Runs the entire machine translation benchmark and prints consolidated results.
"""

import sys
import subprocess

def run_step(step_num, title, script_path):
    print("\n" + "=" * 75)
    print(f"STEP {step_num}: {title.upper()}")
    print("=" * 75)
    cmd = [sys.executable, script_path]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Warning: Step {step_num} exited with status {result.returncode}")

def main():
    print("*" * 75)
    print("FOUNDATIONS OF GENERATIVE AI (ICT4403) - ASSIGNMENT 3 BENCHMARK RUNNER")
    print("*" * 75)

    # 1. Dataset Preprocessing
    run_step(1, "Dataset Acquisition & Preprocessing", "scripts/01_download_and_preprocess.py")

    # 2. Basic Transformer Evaluation
    run_step(2, "Basic Transformer Evaluation (Encoder-Decoder)", "scripts/02_train_basic_transformer.py")

    # 3. mT5 Evaluation
    run_step(3, "mT5 Fine-Tuning & Evaluation (Encoder-Decoder)", "scripts/03_finetune_mt5.py")

    # 4. LLaMA & Gemma Prompting Evaluation
    run_step(4, "LLaMA & Gemma Evaluation (Decoder-Only)", "scripts/04_evaluate_llama_gemma.py")

    # 5. Benchmark Synthesis & Plots
    run_step(5, "Consolidated Benchmark & Comparison Plot Generation", "scripts/05_benchmark_all_models.py")

    # 6. Sampling Temperature Analysis
    run_step(6, "Sampling-Temperature Analysis (T = 0.0 to 1.2)", "scripts/06_sampling_temperature_analysis.py")

    # 7. Responsible & Ethical AI Analysis
    run_step(7, "Responsible & Ethical AI Evaluation", "scripts/07_ethical_bias_analysis.py")

    # 8. Interactive Sample Viewer
    run_step(8, "Displaying Sample Translations Across All 4 Models", "scripts/view_sample_translations.py")

    print("\n" + "*" * 75)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("Artifacts generated:")
    print(" - Results CSVs: results/")
    print(" - Comparison Plots: results/plots/")
    print(" - Word Submission Doc: YourName_RegNum_FGAI_Assignment3.docx")
    print(" - Jupyter Notebook: machine_translation_benchmark.ipynb")
    print("*" * 75)

if __name__ == "__main__":
    main()
