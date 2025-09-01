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
train_size = int(0.8 * len(df))  # 80% para train
test_size = len(df) - train_size  # 20% para test

# Dividir usando slicing y hacer nuevos datasets 
df_train = df[:train_size]     
df_test = df[train_size:]      

# Verificar la división
print(f"Dataset original: {len(df)} registros")
print(f"Train set: {len(df_train)} registros ({len(df_train)/len(df)*100:.1f}%)")
print(f"Test set: {len(df_test)} registros ({len(df_test)/len(df)*100:.1f}%)")

# Agrega bias a los dataframes de train y test
df_train['bias'] = 1
df_test['bias'] = 1

# Reordenar columnas para un mejor acceso en estos datasets
columnas = ['bias', '6000_normalized', '256.1_normalized', '198_normalized']
df_train = df_train[columnas]
df_test = df_test[columnas]

# FUNCIÓN PRINCIPAL DEL CÁLCULO DE GRADIENTE 
def descenso_gradiente_multivariable(X, y, theta, alpha, epochs, tolerancia=1e-6):
    """
    Descenso de gradiente para regresión lineal múltiple
    X: matriz con bias y features normalizados (m x n)
    y: target normalizado (m x 1)
    theta: vector de parámetros iniciales (n x 1)
    """
    m = len(y)
    n = len(theta)
    historial_theta = [theta.copy()]
    historial_error = []

    print(f"Iniciando descenso de gradiente para {n} parámetros")
    print(f"Muestras: {m}, Tasa aprendizaje: {alpha}, Épocas máximas: {epochs}")
    print()

    for epoca in range(epochs):
        # Calcular predicciones manualmente (sin np.dot)
        predicciones = np.zeros(m)
        for i in range(m):
            suma_pred = 0.0
            for j in range(n):
                suma_pred += X[i, j] * theta[j]
            predicciones[i] = suma_pred

        # Calcular errores
        errores = predicciones - y

        # Calcular error cuadrático medio manualmente
        suma_errores_cuad = 0.0
        for error in errores:
            suma_errores_cuad += error ** 2
        error_mse = suma_errores_cuad / m
        historial_error.append(error_mse)

        # Calcular gradientes manualmente (sin np.dot o X.T)
        gradientes = np.zeros(n)
        for j in range(n):  # Para cada parámetro theta[j]
            suma_grad = 0.0
            for i in range(m):  # Para cada muestra
                suma_grad += X[i, j] * errores[i]
            gradientes[j] = (1/m) * suma_grad

        # Guardar theta anterior para convergencia
        theta_anterior = theta.copy()

        # Actualizar parámetros manualmente
        for j in range(n):
            theta[j] = theta[j] - alpha * gradientes[j]
        
        historial_theta.append(theta.copy())

        # Verificar convergencia
        cambio_max = 0.0
        for j in range(n):
            cambio = abs(theta[j] - theta_anterior[j])
            if cambio > cambio_max:
                cambio_max = cambio

        # Imprimir la info actual cada 100 epocas
        if epoca % 100 == 0:
            print(f"Época {epoca + 1}:")
            print(f"  theta = {[f'{t:.6f}' for t in theta]}")
            print(f"  Error MSE = {error_mse:.6f}")
            print(f"  Cambio max theta = {cambio_max:.8f}")
            print()

        # Detener el modelo si llega a la convergencia
        if cambio_max < tolerancia:
            print(f"¡Convergencia en época {epoca + 1}!")
            break
    
    # Si no llegó, regresa a los valores de la época 1000
    return historial_theta, historial_error

# Función de predicciones
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


# CÓDIGO PRINCIPAL

# Extraer arrays con numpy manualmente
X_train = np.column_stack([
    np.ones(len(df_train)),  # Columna de bias (1's)
    df_train['6000_normalized'].values,
    df_train['256.1_normalized'].values
])

y_train = df_train['198_normalized'].values

X_test = np.column_stack([
    np.ones(len(df_test)),  # Columna de bias (1's)
    df_test['6000_normalized'].values,
    df_test['256.1_normalized'].values
])

y_test = df_test['198_normalized'].values

# Parámetros iniciales (theta0 = bias, theta1 = MMAX, theta2 = CACH)
theta_inicial = np.array([0.0, 0.0, 0.0])  # [bias, coef_MMAX, coef_CACH]

# Hiperparámetros
tasa_aprendizaje = 0.01
epocas_maximas = 1000
tolerancia = 1e-6

print("=" * 60)
print("REGRESIÓN LINEAL MÚLTIPLE - DESCENSO DE GRADIENTE MANUAL")
print("=" * 60)
print(f"Dimensiones - X_train: {X_train.shape}, y_train: {y_train.shape}")
print()

# Entrenar modelo
historial_theta, historial_error = descenso_gradiente_multivariable(
    X_train, y_train, theta_inicial, tasa_aprendizaje, epocas_maximas, tolerancia
)

# Parámetros finales
theta_final = historial_theta[-1]
error_final = historial_error[-1] if historial_error else None

print("RESULTADOS FINALES:")
print("=" * 40)
print(f"Épocas ejecutadas: {len(historial_theta) - 1}")
print(f"Parámetros finales:")
print(f"  Bias (θ₀) = {theta_final[0]:.6f}")
print(f"  Coef MMAX (θ₁) = {theta_final[1]:.6f}")
print(f"  Coef CACH (θ₂) = {theta_final[2]:.6f}")
print(f"Error MSE final = {error_final:.6f}")
print()

# Predecir en training set 
y_pred_train = predecir(X_train, theta_final)
suma_error_train = 0.0
for i in range(len(y_train)):
    suma_error_train += (y_pred_train[i] - y_train[i]) ** 2
error_train = suma_error_train / len(y_train)

# Predecir en test set 
y_pred_test = predecir(X_test, theta_final)
suma_error_test = 0.0
for i in range(len(y_test)):
    suma_error_test += (y_pred_test[i] - y_test[i]) ** 2
error_test = suma_error_test / len(y_test)

print("EVALUACIÓN DEL MODELO:")
print("=" * 40)
print(f"Error MSE training: {error_train:.6f}")
print(f"Error MSE test: {error_test:.6f}")
print()

# Calcular R² 
media_y = np.mean(y_train)
ss_total = 0.0
ss_residual = 0.0
for i in range(len(y_train)):
    ss_total += (y_train[i] - media_y) ** 2
    ss_residual += (y_train[i] - y_pred_train[i]) ** 2
r2 = 1 - (ss_residual / ss_total)

print(f"R² en training: {r2:.6f}")
print()

# Ejemplo de predicción para nuevos valores (hardcodeados)
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

# Normalizar 
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

# Imprimir el error en unidades originales:
error_train_real = np.sqrt(error_train) * 100
error_test_real = np.sqrt(error_test) * 100
print(error_test_real)

# Crear el plot donde se ven los valores de las predicciones contra los valores reales:

# Obtener valores reales y predichos en escala ORIGINAL
y_test_real = y_test * std_prp + media_prp  # True values (desnormalizados)
y_pred_test_real = predecir(X_test, theta_final) * std_prp + media_prp  # Predicted values

# Obtener el número de muestras en test
n_real = len(y_test_real)  
indices = np.arange(n_real)  

print(f"Tamaño del test set: {n_real} muestras")

# Crear el gráfico con el tamaño correcto (me dio error al principio si no hacia esto jaja)
plt.figure(figsize=(14, 7))

# Graficar valores reales (azul)
plt.plot(indices, y_test_real,
         'b-', linewidth=2, marker='o', markersize=4,
         label='Valores reales (y)', alpha=0.8)

# Graficar valores predichos (rojo)
plt.plot(indices, y_pred_test_real,
         'r-', linewidth=2, marker='s', markersize=4,
         label='Predicciones (ŷ)', alpha=0.8)

# Personalizar el gráfico 
plt.xlabel('Índice del ejemplo', fontsize=12, fontweight='bold')
plt.ylabel('Valor de PRP', fontsize=12, fontweight='bold')
plt.title(f'Predicciones (rojo) vs Valores reales (azul) - {n_real} ejemplos',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Añadir leyenda
plt.legend(loc='upper right', fontsize=11)

plt.tight_layout()
plt.show()