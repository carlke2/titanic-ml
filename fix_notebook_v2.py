import json, sys, os

NB_PATH = os.path.join("notebooks", "01_baseline.ipynb")

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

def src(cell):
    return "".join(cell["source"])

def set_src(cell, text):
    cell["source"] = text.splitlines(True)

for cell in cells:
    s = src(cell)
    if 'PROJECT_ROOT = os.path.abspath("..")' in s:
        new_src = s.replace(
            'PROJECT_ROOT = os.path.abspath("..")',
            'if os.path.basename(os.getcwd()) == "notebooks":\n    PROJECT_ROOT = os.path.abspath("..")\nelse:\n    PROJECT_ROOT = os.path.abspath(".")'
        )
        set_src(cell, new_src)
        print("[FIXED] PROJECT_ROOT path resolution")
        break
else:
    print("[SKIP] PROJECT_ROOT definition not found or already fixed")

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"[SAVED] {NB_PATH}")
