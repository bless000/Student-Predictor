"""
run_all.py — Runs the complete ML pipeline in order.
After this finishes, launch the app with:
    streamlit run app.py
"""
import subprocess, sys, os

BASE  = os.path.dirname(os.path.abspath(__file__))
steps = [
    ("Step 1/3 — Generating dataset (1,500 students)",      ["python", "src/generate_data.py"]),
    ("Step 2/3 — EDA, Cleaning & Feature Engineering",      ["python", "src/preprocess.py"]),
    ("Step 3/3 — Training 5 ML models & saving best",       ["python", "src/train.py"]),
]

print("=" * 62)
print("   🎓  AcadPredict — Machine Learning Pipeline")
print("=" * 62)

for label, cmd in steps:
    print(f"\n  ▶  {label}")
    print("  " + "─" * 58)
    result = subprocess.run(cmd, cwd=BASE)
    if result.returncode != 0:
        print(f"\n  ✗  Failed at: {label}")
        sys.exit(1)

print("\n" + "=" * 62)
print("  ✅  Pipeline complete!")
print()
print("  🚀  Now launch the web app:")
print("      streamlit run app.py")
print("=" * 62)
