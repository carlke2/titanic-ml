# 🚢 Titanic ML — Machine Learning from Disaster

> A beginner-friendly machine learning project built on the classic
> [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic).

---

## 📖 About the Competition

The **Titanic - Machine Learning from Disaster** competition is Kaggle's
introductory binary classification challenge.

The goal is simple:  
**Predict which passengers survived the sinking of the RMS Titanic.**

You are given information about each passenger (age, sex, ticket class, fare, etc.)
and you must build a model that outputs `1` (survived) or `0` (did not survive).

It is the perfect starting point for anyone learning supervised machine learning.

---

## 🎯 Project Goal

Build a clean, reproducible Python pipeline that:

1. Loads and explores the Titanic dataset
2. Cleans and preprocesses the data
3. Engineers useful features
4. Trains a baseline classifier with scikit-learn
5. Validates the model with cross-validation
6. Generates a `submission.csv` ready to upload to Kaggle

---

## 📁 Project Structure

```
titanic-ml/
│
├── data/                        # Raw Kaggle data files (download from Kaggle)
│   ├── train.csv                #   Training set (891 rows, labelled)
│   ├── test.csv                 #   Test set (418 rows, unlabelled)
│   └── gender_submission.csv    #   Sample submission format
│
├── notebooks/
│   └── 01_baseline.ipynb        # Main experimentation notebook
│
├── submissions/                 # Generated submission CSV files go here
│   └── .gitkeep
│
├── src/                         # Reusable Python modules
│   ├── __init__.py
│   ├── helpers.py               #   General utility functions
│   ├── preprocessing.py         #   Data cleaning & missing value handling
│   ├── features.py              #   Feature engineering
│   ├── model.py                 #   Model training & prediction
│   └── evaluate.py              #   Model evaluation & metrics
│
├── reports/                     # Saved figures, HTML reports
│   └── .gitkeep
│
├── .gitignore
├── .env.example                 # Template for environment variables
├── README.md                    # This file
└── requirements.txt             # Pinned Python dependencies
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites

Make sure you have the following installed:

- [Python 3.10+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)
- [VS Code](https://code.visualstudio.com/) with the **Python** and **Jupyter** extensions

---

### 2. Clone / Open the Project

If you cloned from Git:

```powershell
git clone <your-repo-url>
cd titanic-ml
```

Or if you created it locally, just open the folder in VS Code:

```powershell
code C:\Users\USER\titanic-ml
```

---

### 3. Create the Virtual Environment

Run this once inside the project folder:

```powershell
python -m venv .venv
```

---

### 4. Activate the Virtual Environment

> **Always activate the virtual environment before running any Python command.**

```powershell
# Windows (PowerShell)
.venv\Scripts\activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

You will see `(.venv)` appear at the start of your terminal prompt.

---

### 5. Install Dependencies

With the virtual environment active:

```powershell
pip install pandas numpy matplotlib seaborn scikit-learn jupyter jupyterlab notebook ipykernel python-dotenv
```

Then freeze the exact versions into `requirements.txt`:

```powershell
pip freeze > requirements.txt
```

To reinstall from `requirements.txt` in the future:

```powershell
pip install -r requirements.txt
```

---

### 6. Register the Jupyter Kernel

This tells Jupyter to use your project's virtual environment as a kernel:

```powershell
python -m ipykernel install --user --name=titanic-ml --display-name "Python (titanic-ml)"
```

---

### 7. Place the Kaggle Data Files

Download the three CSV files from Kaggle:

👉 [https://www.kaggle.com/competitions/titanic/data](https://www.kaggle.com/competitions/titanic/data)

Place them in the `data/` folder:

```
data/
├── train.csv
├── test.csv
└── gender_submission.csv
```

> **Note:** These files are tracked by Git. Do **not** place `.zip` files in the
> `data/` folder — they are ignored by `.gitignore`.

---

### 8. Open the Notebook in VS Code

```powershell
# Option A — Open VS Code, then open the notebook from the Explorer panel
code .

# Option B — Launch JupyterLab in the browser
jupyter lab
```

In VS Code, open `notebooks/01_baseline.ipynb`.  
When prompted to select a kernel, choose **Python (titanic-ml)**.

---

## 🔄 Future Workflow

Once the foundation is in place, the development workflow will follow these steps:

| Step | Description |
|------|-------------|
| **1. Load Data** | Read `train.csv` and `test.csv` with pandas |
| **2. Explore Data** | Check shapes, data types, missing values, distributions |
| **3. Clean Missing Values** | Impute Age, fill Embarked, drop Cabin |
| **4. Encode Categorical Columns** | Convert Sex and Embarked to numbers |
| **5. Feature Engineering** | Create FamilySize, IsAlone, Title, AgeBand |
| **6. Train Baseline Model** | Fit a LogisticRegression or RandomForestClassifier |
| **7. Validate Model** | Cross-validation accuracy + confusion matrix |
| **8. Generate submission.csv** | Predict on `test.csv`, save to `submissions/` |
| **9. Upload to Kaggle** | Submit via the Kaggle website or CLI |

---

## ✅ Foundation Checklist

Use this checklist to confirm your environment is fully set up:

- [ ] Project folder created (`titanic-ml/`)
- [ ] Virtual environment created (`.venv/`)
- [ ] Virtual environment activated (`(.venv)` visible in terminal)
- [ ] Packages installed (`pip install ...`)
- [ ] `requirements.txt` generated (`pip freeze > requirements.txt`)
- [ ] Kaggle files placed in `data/` (`train.csv`, `test.csv`, `gender_submission.csv`)
- [ ] Jupyter kernel created (`python -m ipykernel install ...`)
- [ ] Baseline notebook created (`notebooks/01_baseline.ipynb`)
- [ ] Git initialized (`git init`)
- [ ] First commit made (`git commit -m "Initialize Titanic ML project foundation"`)

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Data visualization (base) |
| `seaborn` | Statistical plots and heatmaps |
| `scikit-learn` | Machine learning models & evaluation |
| `jupyter` / `jupyterlab` | Notebook interface |
| `ipykernel` | Register custom Jupyter kernel |
| `python-dotenv` | Load environment variables from `.env` |

---

## 🤝 Contributing

This is a personal learning project. Feel free to fork it and experiment!

---

*Built as a beginner-friendly foundation for the Kaggle Titanic competition.*
