"""
Fix 01_baseline.ipynb:
1. Cell 2 (Load Data): use PROJECT_ROOT-based paths instead of fragile relative paths
2. Cell 3 (EDA bar plots): fix seaborn FutureWarning (palette without hue)
3. Clear all outputs so the notebook runs clean
"""
import json, sys, os

NB_PATH = os.path.join("notebooks", "01_baseline.ipynb")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ── helpers ──────────────────────────────────────────────────────────────────
def src(cell):
    """Return cell source as a single string."""
    return "".join(cell["source"])

def set_src(cell, text):
    """Replace cell source (stored as list of lines)."""
    cell["source"] = text.splitlines(True)

def clear_outputs(cell):
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

# ── Fix 1: Cell 2 – Load Data paths ─────────────────────────────────────────
# Find the cell whose source starts with 'train_raw = load_data'
for cell in cells:
    s = src(cell)
    if 'load_data("../data/train.csv")' in s:
        new_src = s.replace(
            'load_data("../data/train.csv")',
            'load_data(os.path.join(PROJECT_ROOT, "data", "train.csv"))'
        ).replace(
            'load_data("../data/test.csv")',
            'load_data(os.path.join(PROJECT_ROOT, "data", "test.csv"))'
        )
        set_src(cell, new_src)
        print("[FIXED] Load Data cell: paths now use PROJECT_ROOT")
        break
else:
    print("[SKIP] Load Data cell not found or already fixed")

# ── Fix 2: EDA barplot – add hue parameter ──────────────────────────────────
for cell in cells:
    s = src(cell)
    # Fix "Survival Rate by Gender" barplot
    if 'sns.barplot(x="Sex", y="Survived", data=train_raw, palette="Blues_d")' in s:
        new_src = s.replace(
            'sns.barplot(x="Sex", y="Survived", data=train_raw, palette="Blues_d")',
            'sns.barplot(x="Sex", y="Survived", hue="Sex", data=train_raw, palette="Blues_d", legend=False)'
        )
        set_src(cell, new_src)
        print("[FIXED] EDA barplot (Sex): added hue param to silence FutureWarning")

    # Fix "Survival Rate by Passenger Class" barplot
    if 'sns.barplot(x="Pclass", y="Survived", data=train_raw, palette="Purples_d")' in s:
        new_src = s.replace(
            'sns.barplot(x="Pclass", y="Survived", data=train_raw, palette="Purples_d")',
            'sns.barplot(x="Pclass", y="Survived", hue="Pclass", data=train_raw, palette="Purples_d", legend=False)'
        )
        set_src(cell, new_src)
        print("[FIXED] EDA barplot (Pclass): added hue param to silence FutureWarning")

# ── Clear all outputs so the notebook runs clean ────────────────────────────
for cell in cells:
    clear_outputs(cell)
print("[DONE] All cell outputs cleared")

# ── Write back ──────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"[SAVED] {NB_PATH}")
