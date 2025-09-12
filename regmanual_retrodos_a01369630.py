import pandas as pd
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Abrir diálogo para seleccionar archivo (no sé por qué no quería abrirlo de una manera más simple)
root = tk.Tk()
root.withdraw()  # Ocultar ventana principal

file_path = filedialog.askopenfilename(
    title="Seleccionar archivo machine.data",
    filetypes=[("Data files", "*.data"), ("All files", "*.*")]
)

if file_path:
    df = pd.read_csv(file_path)
    print(df.head())
else:
    print("No se seleccionó ningún archivo")
    
df.dropna(inplace=True) # Quitar valores nulos

# Droppear las columnas que no son parámetros para el modelo
df.drop('125', axis=1, inplace=True)
df.drop('256', axis=1, inplace=True)
df.drop('16', axis=1, inplace=True)
df.drop('128', axis=1, inplace=True)
df.drop('-adviser', axis=1, inplace=True)
df.drop('32/60', axis=1, inplace=True)
df.drop('199', axis=1, inplace=True)

# Normalizar las columnas que si se van a usar 
mean_6000 = df['6000'].mean()
std_6000 = df['6000'].std()
df['6000_normalized'] = (df['6000'] - mean_6000) / std_6000

mean_256_1 = df['256.1'].mean()
std_256_1 = df['256.1'].std()
df['256.1_normalized'] = (df['256.1'] - mean_256_1) / std_256_1

# Normalize '198' column
mean_198 = df['198'].mean()
std_198 = df['198'].std()
df['198_normalized'] = (df['198'] - mean_198) / std_198

# Plot para visualizar los scattered plots de los datos normalizados
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=df['6000_normalized'], y=df['198_normalized'])
plt.title('6000 Normalized vs 198 Normalized (z)')
plt.xlabel('6000 Normalized')
plt.ylabel('198_normalized')

plt.subplot(1, 2, 2)
sns.scatterplot(x=df['256.1_normalized'], y=df['198_normalized'])
plt.title('256.1 Normalized vs 198 Normalized (z)')
plt.xlabel('256.1 Normalized')
plt.ylabel('198_normalized')

plt.tight_layout()
plt.show()

# Mezclar el DataFrame aleatoriamente
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calcular tamaños
total_size = len(df)
train_size = int(0.6 * total_size)  # 60%
val_size = int(0.2 * total_size)    # 20%
test_size = total_size - train_size - val_size  # 20%

# Dividir usando slicing directo
df_train = df[:train_size]                      # Primer 60%
df_val = df[train_size:train_size + val_size]   # Segundo 20%
df_test = df[train_size + val_size:]            # Tercero 20%

# Verificar la división
print(f"Dataset original: {total_size} registros")
print(f"Train set: {len(df_train)} registros ({len(df_train)/total_size*100:.1f}%)")
print(f"Validation set: {len(df_val)} registros ({len(df_val)/total_size*100:.1f}%)")
print(f"Test set: {len(df_test)} registros ({len(df_test)/total_size*100:.1f}%)")

# Agrega bias a tus DataFrames de train validation y test
df_train['bias'] = 1
df_val['bias'] = 1
df_test['bias'] = 1

# Reordena columnas para un mejor acceso en estos datasets
columnas = ['bias', '6000_normalized', '256.1_normalized', '198_normalized']
df_train = df_train[columnas]
df_val = df_val[columnas]
df_test = df_test[columnas]

# FUNCIÓN PRINCIPAL DEL CÁLCULO DE GRADIENTE 
def descenso_gradiente_multivariable(X_train, y_train, X_val, y_val, theta, alpha, epochs, tolerancia=1e-6):
    """
    Descenso de gradiente para regresión lineal múltiple con validation
    X_train: matriz de entrenamiento con bias y features normalizados (m x n)
    y_train: target normalizado de entrenamiento (m x 1)
    X_val: matriz de validation con bias y features normalizados (k x n)
    y_val: target normalizado de validation (k x 1)
    theta: vector de parámetros iniciales (n x 1)
    """
    m = len(y_train)
    n = len(theta)
    historial_theta = [theta.copy()]
    historial_error_train = []
    historial_error_val = []  # Nuevo: historial de error de validation

    print(f"Iniciando descenso de gradiente para {n} parámetros")
    print(f"Muestras training: {m}, Muestras validation: {len(y_val)}")
    print(f"Tasa aprendizaje: {alpha}, Épocas máximas: {epochs}")
    print()

    for epoca in range(epochs):
        # 1. Calcular predicciones manualmente (TRAIN)
        predicciones_train = np.zeros(m)
        for i in range(m):
            suma_pred = 0.0
            for j in range(n):
                suma_pred += X_train[i, j] * theta[j]
            predicciones_train[i] = suma_pred

        # 2. Calcular errores (TRAIN)
        errores_train = predicciones_train - y_train

        # 3. Calcular error cuadrático medio manualmente (TRAIN)
        suma_errores_cuad = 0.0
        for error in errores_train:
            suma_errores_cuad += error ** 2
        error_mse_train = suma_errores_cuad / m
        historial_error_train.append(error_mse_train)

        # 4. Calcular error en VALIDATION
        predicciones_val = np.zeros(len(y_val))
        for i in range(len(y_val)):
            suma_pred = 0.0
            for j in range(n):
                suma_pred += X_val[i, j] * theta[j]
            predicciones_val[i] = suma_pred

        error_mse_val = np.mean((predicciones_val - y_val)**2)
        historial_error_val.append(error_mse_val)

        # 5. Calcular gradientes manualmente (usando TRAIN)
        gradientes = np.zeros(n)
        for j in range(n):  # Para cada parámetro theta[j]
            suma_grad = 0.0
            for i in range(m):  # Para cada muestra de training
                suma_grad += X_train[i, j] * errores_train[i]
            gradientes[j] = (1/m) * suma_grad

        # 6. Guardar theta anterior para convergencia
        theta_anterior = theta.copy()

        # 7. Actualizar parámetros manualmente
        for j in range(n):
            theta[j] = theta[j] - alpha * gradientes[j]

        historial_theta.append(theta.copy())

        # 8. Verificar convergencia
        cambio_max = 0.0
        for j in range(n):
            cambio = abs(theta[j] - theta_anterior[j])
            if cambio > cambio_max:
                cambio_max = cambio

        # 9. Early stopping basado en validation error
        if epoca > 20 and historial_error_val[-1] > np.mean(historial_error_val[-5:]):
            print(f"Early stopping en época {epoca + 1} - Validation error aumentando")
            break

        if epoca % 100 == 0 or epoca < 5:
            print(f"Época {epoca + 1}:")
            print(f"  theta = {[f'{t:.6f}' for t in theta]}")
            print(f"  Error MSE Train = {error_mse_train:.6f}")
            print(f"  Error MSE Val = {error_mse_val:.6f}")  
            print(f"  Cambio max theta = {cambio_max:.8f}")
            print()

        if cambio_max < tolerancia:
            print(f"¡Convergencia en época {epoca + 1}!")
            break

    return historial_theta, historial_error_train, historial_error_val

def predecir(X, theta):
    """Predecir valores para nuevas muestras manualmente"""
    if len(X.shape) == 1:
        # Si es un solo vector [1, x1, x2]
        prediccion = 0.0
        for j in range(len(theta)):
            prediccion += X[j] * theta[j]
        return prediccion
    else:
        # Si es matriz de múltiples muestras
        predicciones = np.zeros(len(X))
        for i in range(len(X)):
            suma = 0.0
            for j in range(len(theta)):
                suma += X[i, j] * theta[j]
            predicciones[i] = suma
        return predicciones

# PREPARACIÓN DE DATOS CON VALIDATION SET (60-20-20)
# Preparar matrices de TRAINING
X_train = np.column_stack([
    np.ones(len(df_train)),  # Columna de bias
    df_train['6000_normalized'].values,
    df_train['256.1_normalized'].values
])
y_train = df_train['198_normalized'].values

# Preparar matrices de VALIDATION 
X_val = np.column_stack([
    np.ones(len(df_val)),  # Columna de bias
    df_val['6000_normalized'].values,
    df_val['256.1_normalized'].values
])
y_val = df_val['198_normalized'].values

# Preparar matrices de TEST
X_test = np.column_stack([
    np.ones(len(df_test)),  # Columna de bias
    df_test['6000_normalized'].values,
    df_test['256.1_normalized'].values
])
y_test = df_test['198_normalized'].values

# Verificar dimensiones
print("=" * 60)
print("DIMENSIONES DE LOS DATASETS")
print("=" * 60)
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")  # Nuevo
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
print()

# Parámetros iniciales (theta0 = bias, theta1 = MMAX, theta2 = CACH)
theta_inicial = np.array([0.0, 0.0, 0.0])  # [bias, coef_MMAX, coef_CACH]

# Hiperparámetros
tasa_aprendizaje = 0.01
epocas_maximas = 10000
tolerancia = 1e-6

print("=" * 60)
print("REGRESIÓN LINEAL MÚLTIPLE - DESCENSO DE GRADIENTE CON VALIDATION")
print("=" * 60)

# Entrenar modelo con validation
historial_theta, historial_error_train, historial_error_val = descenso_gradiente_multivariable(
    X_train, y_train, X_val, y_val, theta_inicial, tasa_aprendizaje, epocas_maximas, tolerancia
)

# Parámetros finales
theta_final = historial_theta[-1]
error_final_train = historial_error_train[-1] if historial_error_train else None
error_final_val = historial_error_val[-1] if historial_error_val else None

print("RESULTADOS FINALES:")
print("=" * 40)
print(f"Épocas ejecutadas: {len(historial_theta) - 1}")
print(f"Parámetros finales:")
print(f"  Bias (θ₀) = {theta_final[0]:.6f}")
print(f"  Coef MMAX (θ₁) = {theta_final[1]:.6f}")
print(f"  Coef CACH (θ₂) = {theta_final[2]:.6f}")
print(f"Error MSE final (Train): {error_final_train:.6f}")
print(f"Error MSE final (Validation): {error_final_val:.6f}")
print()

# EVALUACIÓN EN LOS TRES CONJUNTOS:
# Predecir en training set
y_pred_train = predecir(X_train, theta_final)
suma_error_train = 0.0
for i in range(len(y_train)):
    suma_error_train += (y_pred_train[i] - y_train[i]) ** 2
error_train = suma_error_train / len(y_train)

# Predecir en validation set
y_pred_val = predecir(X_val, theta_final)
suma_error_val = 0.0
for i in range(len(y_val)):
    suma_error_val += (y_pred_val[i] - y_val[i]) ** 2
error_val = suma_error_val / len(y_val)

# Predecir en test set
y_pred_test = predecir(X_test, theta_final)
suma_error_test = 0.0
for i in range(len(y_test)):
    suma_error_test += (y_pred_test[i] - y_test[i]) ** 2
error_test = suma_error_test / len(y_test)

print("EVALUACIÓN COMPLETA DEL MODELO:")
print("=" * 50)
print(f"Error MSE training: {error_train:.6f}")
print(f"Error MSE validation: {error_val:.6f}")
print(f"Error MSE test: {error_test:.6f}")
print()

# Calcular R² para todos los conjuntos
def calcular_r2(y_real, y_pred):
    media_y = np.mean(y_real)
    ss_total = 0.0
    ss_residual = 0.0
    for i in range(len(y_real)):
        ss_total += (y_real[i] - media_y) ** 2
        ss_residual += (y_real[i] - y_pred[i]) ** 2
    return 1 - (ss_residual / ss_total)

r2_train = calcular_r2(y_train, y_pred_train)
r2_val = calcular_r2(y_val, y_pred_val)  # Nuevo
r2_test = calcular_r2(y_test, y_pred_test)  # Nuevo

print("R² EN TODOS LOS CONJUNTOS:")
print("=" * 30)
print(f"R² training: {r2_train:.6f}")
print(f"R² validation: {r2_val:.6f}")  # Nuevo
print(f"R² test: {r2_test:.6f}")  # Nuevo
print()

# EJEMPLO DE PREDICCIÓN

print("EJEMPLO DE PREDICCIÓN:")
print("=" * 40)

# Valores nuevos (en escala original)
media_mmax = mean_6000
std_mmax = std_6000

media_cach = mean_256_1
std_cach = std_256_1

media_prp = mean_198
std_prp = std_198

# Valores de ejemplo (en escala original)
nuevo_mmax = 8000  # KB
nuevo_cach = 512   # KB

# Normalizar manualmente
mmax_norm = (nuevo_mmax - media_mmax) / std_mmax
cach_norm = (nuevo_cach - media_cach) / std_cach

# Crear vector de features con bias
X_nuevo = np.array([1, mmax_norm, cach_norm])

# Predecir (valor normalizado)
prp_pred_norm = 0.0
for j in range(len(theta_final)):
    prp_pred_norm += X_nuevo[j] * theta_final[j]

# Revertir normalización
prp_pred = prp_pred_norm * std_prp + media_prp

print(f"Para MMAX={nuevo_mmax}, CACH={nuevo_cach}:")
print(f"PRP predicho: {prp_pred:.2f}")