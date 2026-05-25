
# %% [code]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import re
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

from _SimpleLinearRegression import SimpleLinearRegression
from _MultipleLinearRegression import MultipleLinearRegression
from _LassoRegression import LassoRegression
from _RidgeRegression import RidgeRegression
from _PolynomialRegression import PolynomialRegression
# %% [code]
# Load dataset
path = "used_cars.csv"  
df = pd.read_csv(path)

print("Initial shape:", df.shape)
print("Columns:", df.columns.tolist())
df.head()

# %% [code]
# --- Clean price ---
df['price_clean'] = (
    df['price']
    .replace('[\$,]', '', regex=True)
    .astype(float)
)

# --- Clean mileage ---
df['milage_clean'] = (
    df['milage']
    .str.replace('mi.', '', regex=False)
    .str.replace(',', '', regex=False)
)
df['milage_clean'] = pd.to_numeric(df['milage_clean'], errors='coerce')

# --- Compute car age ---
CURRENT_YEAR = 2025
df['car_age'] = CURRENT_YEAR - df['model_year']

# --- Parse engine size (in Liters) ---
def extract_engine_liters(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r'(\d+\.\d+|\d+)\s*[lL]', str(x))
    return float(m.group(1)) if m else np.nan

df['engine_liters'] = df['engine'].apply(extract_engine_liters)
df['engine_liters'].fillna(df['engine_liters'].median(), inplace=True)

# --- Remove impossible values ---
df = df[(df['price_clean'] > 1000) & (df['price_clean'] < 200000)]
df = df[(df['milage_clean'] > 0) & (df['milage_clean'] < 300000)]

print("After cleaning:", df.shape)

# %% [code]
# --- Accident & title flags ---
df['had_accident'] = df['accident'].apply(lambda x: 0 if str(x).lower().startswith('none') else 1)
df['clean_title_flag'] = df['clean_title'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

# --- Encode brand and model (reduce categories) ---
top_brands = df['brand'].value_counts().nlargest(10).index
df['brand_top'] = df['brand'].apply(lambda x: x if x in top_brands else 'Other')

top_models = df['model'].value_counts().nlargest(20).index
df['model_top'] = df['model'].apply(lambda x: x if x in top_models else 'Other')

# --- Encode transmission and fuel type ---
df['transmission'] = df['transmission'].fillna('Unknown')
df['fuel_type'] = df['fuel_type'].fillna('Unknown')

# --- Interaction features ---
df['age_x_mileage'] = df['car_age'] * df['milage_clean']
df['age_x_engine'] = df['car_age'] * df['engine_liters']

# --- Remove outliers ---
q1, q99 = df['price_clean'].quantile([0.01, 0.99])
df = df[(df['price_clean'] > q1) & (df['price_clean'] < q99)]

# --- Log-transform the target ---
df['log_price'] = np.log1p(df['price_clean'])

print("After feature engineering:", df.shape)

# %% [code]
# Create dummy variables
df = pd.get_dummies(
    df,
    columns=['brand_top', 'model_top', 'transmission', 'fuel_type'],
    drop_first=True
)

print("Shape after encoding:", df.shape)

# %% [code]
# Select ONLY basic features (minimal feature set for lower performance)

basic_features = [
    'car_age',
    'milage_clean',
    'engine_liters'
]

X = df[basic_features].values
y = df['log_price'].values  # log-transformed price

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print("Train shape:", X_train.shape)
print("Features used:", basic_features)

# %% [code]
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)


# Train on mileage only
X_train_slr = X_train[:, 1].reshape(-1, 1)
X_val_slr = X_val[:, 1].reshape(-1, 1)

slr = SimpleLinearRegression()
slr.fit(X_train_slr, y_train)
y_pred_slr = slr.predict(X_val_slr)

# %% [code]


# Train and predict
mlr = MultipleLinearRegression()
mlr.fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_val)

# %% [code]

ridge = RidgeRegression(alpha=0.5)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_val)

# %% [code]


# Train and predict
pr = PolynomialRegression(degree=2)
pr.fit(X_train, y_train)
y_pred_pr = pr.predict(X_val)


lasso = LassoRegression(alpha=0.05, n_iter=1000, lr=0.01)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_val)

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

results_df = pd.DataFrame(results, columns=["Model", "MSE", "RMSE", "R²"]).sort_values(by="R²", ascending=False)
print("\n🏆 Best Model:", results_df.iloc[0]["Model"])
results_df

plt.figure(figsize=(6,6))
plt.scatter(y_val, y_pred_mlr, alpha=0.4, color='royalblue')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("Actual Log(Price)")
plt.ylabel("Predicted Log(Price)")
plt.title("Actual vs Predicted (MLR)")
plt.grid(True)
plt.show()