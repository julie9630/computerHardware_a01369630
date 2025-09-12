import xgboost as xgb
import numpy as np
import pandas as pd
from tkinter import filedialog
from sklearn.metrics import mean_squared_error, r2_score

# Subir el archivo .data
file_path = filedialog.askopenfilename(
    title="Seleccionar archivo machine.data",
    filetypes=[("Data files", "*.data"), ("All files", "*.*")])

# Cargar el archivo a un dataset (le puse las columnas de los nombres que salen en la parte de MACHINE.NAMES)
df = pd.read_csv(file_path, names=['VendorName', 'ModelName', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'])

# Procesar el data (droppear las columnas que no voy a usar)
df.drop(['VendorName', 'ModelName', 'ERP', 'MYCT', 'MMIN', 'CHMIN', 'CHMAX' ], axis=1, inplace=True)
df.dropna(inplace=True)

# Normalizar las columnas que voy a usar (con z score otra vez)
mean_6000 = df['MMAX'].mean()
std_6000 = df['MMAX'].std()
df['6000_normalized'] = (df['MMAX'] - mean_6000) / std_6000

mean_256_1 = df['CACH'].mean()
std_256_1 = df['CACH'].std()
df['256.1_normalized'] = (df['CACH'] - mean_256_1) / std_256_1

mean_198 = df['PRP'].mean()
std_198 = df['PRP'].std()
df['198_normalized'] = (df['PRP'] - mean_198) / std_198

# Mezclar el DataFrame aleatoriamente
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calcular tamaños para cada datasets
total_size = len(df)
train_size = int(0.6 * total_size)  # 60%
val_size = int(0.2 * total_size)    # 20%
test_size = total_size - train_size - val_size  # 20%

# Dividir usando slicing
df_train = df[:train_size].copy()
df_val = df[train_size:train_size + val_size].copy()
df_test = df[train_size + val_size:].copy()

# Preparar datos para XGBoost
X_train = np.column_stack([
    np.ones(len(df_train)),
    df_train['6000_normalized'].values,
    df_train['256.1_normalized'].values
])
y_train = df_train['198_normalized'].values

X_val = np.column_stack([
    np.ones(len(df_val)),
    df_val['6000_normalized'].values,
    df_val['256.1_normalized'].values
])
y_val = df_val['198_normalized'].values

X_test = np.column_stack([
    np.ones(len(df_test)),
    df_test['6000_normalized'].values,
    df_test['256.1_normalized'].values
])
y_test = df_test['198_normalized'].values

# Crear matrices DMatrix para XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Configurar parámetros del modelo
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.1,
    'max_depth': 6
}

num_boost_round = 1000

print("XGBoost parameters and number of boosting rounds have been set.")
print("Parameters:", params)
print("Number of boosting rounds:", num_boost_round)

# Entrenar el modelo
evals = [(dtrain, 'train'), (dval, 'validation')]

model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=True
)

# Hacer predicciones en el dataset de test
y_pred_test = model.predict(dtest)

# Calcular el MSE 
mse_test = mean_squared_error(y_test, y_pred_test)

# Calcular el R**2
r2_test = r2_score(y_test, y_pred_test)

# Imprimir los resultados
print(f"Test Set MSE: {mse_test:.6f}")
print(f"Test Set R-squared: {r2_test:.6f}")