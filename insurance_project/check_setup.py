"""
================================================================================
SETUP VERIFICATION SCRIPT
================================================================================
Run this BEFORE train_model.py if you're unsure your environment is set up
correctly, or if you hit a "ModuleNotFoundError" for any package.

Usage:
    python check_setup.py
    (or python3 check_setup.py — see note on python vs python3 below)
================================================================================
"""

import sys
import importlib

print("=" * 70)
print("ENVIRONMENT CHECK")
print("=" * 70)
print(f"Python executable in use: {sys.executable}")
print(f"Python version:           {sys.version.split()[0]}")
print()
print("This is the EXACT Python that must have the packages installed.")
print("If you installed packages with 'pip install' but this path above")
print("looks different from where pip installed to, that mismatch is the")
print("most common cause of 'ModuleNotFoundError' after a successful install.")
print("=" * 70)
print()

required_packages = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "shap": "shap",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "joblib": "joblib",
    "streamlit": "streamlit",
    "plotly": "plotly",
}

missing = []
print("Checking required packages...\n")
for import_name, pip_name in required_packages.items():
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown version")
        print(f"  [OK]      {pip_name:<15} {version}")
    except ImportError:
        print(f"  [MISSING] {pip_name:<15} NOT INSTALLED for this Python")
        missing.append(pip_name)

print()
print("=" * 70)
if missing:
    print(f"RESULT: {len(missing)} package(s) missing for THIS Python interpreter.")
    print()
    print("Fix — run this EXACT command (same Python that ran this check):")
    print(f"    {sys.executable} -m pip install {' '.join(missing)}")
    print()
    print("If you already ran 'pip install -r requirements.txt' and still see")
    print("this, it means 'pip' on your system points to a DIFFERENT Python")
    print("than the one shown above. Use this safer pattern from now on:")
    print(f"    {sys.executable} -m pip install -r requirements.txt")
    print(f"    {sys.executable} train_model.py")
    print(f"    {sys.executable} -m streamlit run app/app.py")
    sys.exit(1)
else:
    print("RESULT: All required packages are correctly installed.")
    print("You're good to run: python train_model.py")
print("=" * 70)
