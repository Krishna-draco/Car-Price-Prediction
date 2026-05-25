# Used Cars Regression — Lightweight ML Pipeline

Project that demonstrates simple custom regression models trained on a `used_cars.csv` dataset. The code implements several regressors (simple linear, multiple linear, polynomial, ridge, lasso) as small, readable Python classes and provides two example runners that show a minimal baseline and a fuller experimental pipeline.

**Contents**

- **`main_1.py`**: Minimal pipeline. Uses a small feature set (`car_age`, `milage_clean`, `engine_liters`), scales features, trains models from the `_*.py` modules, evaluates (MSE / RMSE / R²) and plots a basic Actual vs Predicted scatter.
- **`main_2.py`**: Full pipeline. Adds additional features (`had_accident`, `clean_title_flag`, interaction terms) and one-hot encodes categorical fields (brand/model/transmission/fuel). Produces richer visualizations (Seaborn bar charts and multiple Actual vs Predicted subplots).
- **Model modules**: `_SimpleLinearRegression.py`, `_MultipleLinearRegression.py`, `_PolynomialRegression.py`, `_RidgeRegression.py`, `_LassoRegression.py` — plain NumPy-based implementations used by the mains.
- **Data**: `used_cars.csv` — dataset expected in the repository root.

**Why two mains?**

- `main_1.py` is a compact baseline for quick experiments and reproducibility.
- `main_2.py` is the extended experiment showing a more realistic feature set and visualizations.

Keep both if you want a clear baseline vs. full experiment comparison. If you prefer a tidier repo, consider refactoring shared preprocessing code into a module (e.g., `preprocessing.py`) and using small runner scripts that import the pipeline with different feature flags.

**Quick start**

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the minimal baseline or full pipeline:

```bash
python main_1.py
python main_2.py
```

Notes: both scripts expect `used_cars.csv` to be present. Visualizations will open using matplotlib; on headless systems you may need to save figures instead of showing them.

**Project notes & recommendations**

- Imports: the main scripts import the model modules with leading underscores (e.g., `_SimpleLinearRegression.py`). If you rename files, keep imports consistent or convert the model files into a package (folder with `__init__.py`).
- Refactor suggestion: extract preprocessing (CSV reading, cleaning, feature engineering, and scaling) into a single `preprocessing.py` so both runners share the same logic and only differ by feature selection. Add `run_baseline.py` and `run_full.py` as thin wrappers.
- Reproducibility: random seed is set in both mains (`np.random.seed(42`)). For more control, consider exposing seeds and hyperparameters via command-line flags or a small config file.

**Files**

- `main_1.py` — baseline runner
- `main_2.py` — full runner with extended features and visualizations
- `_SimpleLinearRegression.py`, `_MultipleLinearRegression.py`, `_PolynomialRegression.py`, `_RidgeRegression.py`, `_LassoRegression.py` — model implementations
- `used_cars.csv` — dataset
- `requirements.txt` — Python dependencies

If you want, I can:

- Refactor preprocessing into a shared module and add two small runner scripts, or
- Archive `main_1.py` to a safe folder if you prefer a single canonical script, or
- Add command-line flags to `main_2.py` to toggle baseline vs full features.

---

Generated on May 25, 2026
