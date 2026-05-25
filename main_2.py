import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score # Note: using custom r2_score defined later
import warnings, re
import seaborn as sns
warnings.filterwarnings("ignore")
np.random.seed(42) # Ensure reproducibility

# --- Import Custom Models (Ensure these files exist in your directory) ---
from _SimpleLinearRegression import SimpleLinearRegression
from _MultipleLinearRegression import MultipleLinearRegression
from _LassoRegression import LassoRegression
from _RidgeRegression import RidgeRegression
from _PolynomialRegression import PolynomialRegression
# ------------------------------------------------------------------------

# %% [code]
# Load dataset
path = "used_cars.csv"  # update if needed
df = pd.read_csv(path)

print("Original shape:", df.shape)
df.head()

# %% [code]
# --- Data Cleaning and Feature Creation ---

# Clean price column
df['price_clean'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

# Clean mileage
df['milage_clean'] = df['milage'].str.replace('mi.', '', regex=False).str.replace(',', '', regex=False)
df['milage_clean'] = pd.to_numeric(df['milage_clean'], errors='coerce')

# Compute car age
CURRENT_YEAR = 2025
df['car_age'] = CURRENT_YEAR - df['model_year']

# Parse engine liters
def extract_engine_liters(x):
    if pd.isna(x): return np.nan
    m = re.search(r'(\d+\.\d+|\d+)\s*[lL]', str(x))
    return float(m.group(1)) if m else np.nan

df['engine_liters'] = df['engine'].apply(extract_engine_liters)
df['engine_liters'].fillna(df['engine_liters'].median(), inplace=True)

# %% [code]
# 1️⃣ Log-transform target (price) and filter impossible values
df = df[(df['price_clean'] > 1000) & (df['price_clean'] < 200000)]
df['log_price'] = np.log1p(df['price_clean'])

# 2️⃣ Add brand & model encoding (reduce cardinality)
top_brands = df['brand'].value_counts().nlargest(10).index
df['brand_top'] = df['brand'].apply(lambda x: x if x in top_brands else 'Other')

top_models = df['model'].value_counts().nlargest(20).index
df['model_top'] = df['model'].apply(lambda x: x if x in top_models else 'Other')

# 3️⃣ Accident & title flags
df['had_accident'] = df['accident'].apply(lambda x: 0 if str(x).lower().startswith('none') else 1)
df['clean_title_flag'] = df['clean_title'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

# 4️⃣ Transmission & fuel type (categorical dummies)
df['transmission'] = df['transmission'].fillna('Unknown')
df['fuel_type'] = df['fuel_type'].fillna('Unknown')

# 5️⃣ Interaction features
df['age_x_mileage'] = df['car_age'] * df['milage_clean']
df['age_x_engine'] = df['car_age'] * df['engine_liters']

# 6️⃣ Remove outliers (based on price_clean)
q1, q99 = df['price_clean'].quantile([0.01, 0.99])
df = df[(df['price_clean'] > q1) & (df['price_clean'] < q99)]

# 7️⃣ Encode categorical features
df = pd.get_dummies(df, columns=['brand_top', 'model_top', 'transmission', 'fuel_type'], drop_first=True)

print("After feature engineering:", df.shape)

# %% [code]
# Select features
features = [
    'car_age', 'milage_clean', 'engine_liters',
    'had_accident', 'clean_title_flag',
    'age_x_mileage', 'age_x_engine'
] + [col for col in df.columns if any(prefix in col for prefix in ['brand_top_', 'model_top_', 'transmission_', 'fuel_type_'])]

X = df[features].values
y = df['log_price'].values  # log-transformed target

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# %% [code]
# --- Custom Metrics Functions ---

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)
# --------------------------------

# Use only mileage feature for SLR (column index 1 in scaled data)
X_train_slr = X_train_scaled[:, 1].reshape(-1, 1)
X_val_slr = X_val_scaled[:, 1].reshape(-1, 1)

# Simple Linear Regression
slr = SimpleLinearRegression()
slr.fit(X_train_slr, y_train)
y_pred_slr = slr.predict(X_val_slr)

# Multiple Linear Regression
mlr = MultipleLinearRegression()
mlr.fit(X_train_scaled, y_train)
y_pred_mlr = mlr.predict(X_val_scaled)

# Polynomial Regression (Degree 2)
pr = PolynomialRegression(degree=2)
pr.fit(X_train_scaled, y_train)
y_pred_pr = pr.predict(X_val_scaled)

# %% [code]
# Ridge Regression
ridge = RidgeRegression(alpha=0.5, lr=0.01, n_iter=1000)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_val_scaled)

# Lasso Regression
lasso = LassoRegression(alpha=0.05, lr=0.01, n_iter=1000)
lasso.fit(X_train_scaled, y_train)
y_pred_lasso = lasso.predict(X_val_scaled)

# %% [code]
# --- Evaluation and Results Table ---

def evaluate(y_true, y_pred, name):
    m = mse(y_true, y_pred)
    r = rmse(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<30} | MSE: {m:<10.4f} | RMSE: {r:<8.4f} | R²: {r2:<8.4f}")
    return [name, m, r, r2]

results = []
results.append(evaluate(y_val, y_pred_slr, "Simple Linear Regression"))
results.append(evaluate(y_val, y_pred_mlr, "Multiple Linear Regression"))
results.append(evaluate(y_val, y_pred_pr, "Polynomial Regression (deg=2)"))
results.append(evaluate(y_val, y_pred_ridge, "Ridge Regression"))
results.append(evaluate(y_val, y_pred_lasso, "Lasso Regression"))

# Create results DataFrame using the calculated values
results_df = pd.DataFrame(results, columns=["Model", "MSE", "RMSE", "R²"]).sort_values(by="R²", ascending=False)
print("\n🏆 Best Model:", results_df.iloc[0]["Model"])
print(results_df) # Print the final table

# --- Visualization of Performance Metrics (Using Calculated results_df) ---

# Melt the DataFrame for Seaborn plotting
df_melted = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

# 1. Separate data for Error Metrics (MSE/RMSE) and R²
df_error = df_melted[df_melted['Metric'].isin(['MSE', 'RMSE'])]
df_r2 = df_melted[df_melted['Metric'] == 'R²']

# 2. Create the figure with two separate subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)


# --- Subplot 1: MSE and RMSE Comparison (Graph 1) ---
sns.barplot(
    x='Model',
    y='Score',
    hue='Metric',
    data=df_error,
    palette={'MSE': '#FF9800', 'RMSE': '#2196F3'},
    ax=axes[0]
)

axes[0].tick_params(axis='x', rotation=45)
axes[0].set_title('Graph 1: Comparison of Error Metrics (MSE & RMSE)', fontsize=14)
axes[0].set_ylabel('Error Score (Log-Price Units)', fontsize=12)
axes[0].set_xlabel('Regression Model', fontsize=12)
axes[0].grid(axis='y', linestyle='--', alpha=0.6)
axes[0].legend(title='Metric', loc='upper right')
axes[0].set_ylim(0, df_error['Score'].max() * 1.1)

# Annotate Error Metrics
for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.4f', padding=3, fontsize=9)


# --- Subplot 2: R² Comparison (Graph 2) ---
sns.barplot(
    x='Model',
    y='Score',
    data=df_r2,
    palette='viridis',
    ax=axes[1],
    dodge=False
)

axes[1].tick_params(axis='x', rotation=45)
axes[1].set_title('Graph 2: Comparison of R² Score', fontsize=14)
axes[1].set_ylabel('R² Score (Variance Explained)', fontsize=12)
axes[1].set_xlabel('Regression Model', fontsize=12)
axes[1].set_ylim(0.0, 1.0) # Set Y-limit to R² range
axes[1].grid(axis='y', linestyle='--', alpha=0.6)

# Add R2 values on the bars for clarity
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.4f}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10, weight='bold')

plt.suptitle('Model Performance Comparison: Separated Metrics (Using Calculated R² Scores)', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %% [code]
# --- Actual vs. Predicted Scatter Plots ---
# 1. Collect all predictions and model names
predictions = {
    "Simple Linear Regression": y_pred_slr,
    "Multiple Linear Regression": y_pred_mlr,
    "Polynomial Regression (deg=2)": y_pred_pr,
    "Ridge Regression": y_pred_ridge,
    "Lasso Regression": y_pred_lasso
}

# Determine a universal min/max range for the plot diagonals
min_val = y_val.min()
max_val = y_val.max()

# 2. Create the plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten() # Flatten the 2x3 grid array for easy indexing

for i, (name, y_pred) in enumerate(predictions.items()):
    ax = axes[i]

    # Scatter plot of actual vs. predicted values
    ax.scatter(y_val, y_pred, alpha=0.5, s=20, color='royalblue')

    # Ideal fit line (y=x diagonal)
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')

    # Set titles and labels
    ax.set_title(f'Actual vs. Predicted: {name}', fontsize=14)
    ax.set_xlabel("Actual Log(Price)", fontsize=11)
    ax.set_ylabel("Predicted Log(Price)", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.6)

    # Optional: Add R² score to the plot
    r2 = r2_score(y_val, y_pred) # Uses the custom R2 function
    ax.text(min_val + 0.5, max_val - 0.2, f'$R^2$: {r2:.4f}',
             fontsize=12, color='red', backgroundcolor='white')

# Hide the last empty subplot (since we have 5 models, not 6)
if len(predictions) < len(axes):
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.show()