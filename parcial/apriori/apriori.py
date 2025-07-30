import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from mlxtend.frequent_patterns import apriori, association_rules
import mlflow
import time
import psutil
import os

# Configurar pandas para mostrar mejor los datos
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Reglas_Apriori")

# =====================
# Cargar el dataset
# =====================
df = pd.read_csv('cleaned_online_retail.csv')

# =====================
# Preprocesamiento adicional para Apriori
# =====================
df['Description'] = df['Description'].str.strip()
df.dropna(subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype(str)
df = df[~df['InvoiceNo'].str.contains('C')]  # Quitar devoluciones
df = df[df['Quantity'] > 0]

# =====================
# Crear matriz de transacciones
# =====================
basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

# =====================
# Monitoreo de recursos
# =====================
process = psutil.Process(os.getpid())
start_time = time.time()

# Memoria antes del algoritmo
memory_before = process.memory_info().rss / (1024 * 1024)  # en MB
print(f"Memoria antes del algoritmo: {memory_before:.2f} MB")

# Variable para rastrear el pico de memoria
peak_memory = memory_before

# =====================
# Apriori y reglas de asociación
# =====================
print("Iniciando algoritmo Apriori...")

# Medir memoria durante frequent_itemsets
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
memory_after_frequent = process.memory_info().rss / (1024 * 1024)
peak_memory = max(peak_memory, memory_after_frequent)
print(f"Memoria después de frequent itemsets: {memory_after_frequent:.2f} MB")

# Medir memoria durante association_rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
memory_after_rules = process.memory_info().rss / (1024 * 1024)
peak_memory = max(peak_memory, memory_after_rules)
print(f"Memoria después de reglas de asociación: {memory_after_rules:.2f} MB")

# =====================
# Calcular métricas adicionales leverage y conviction
# =====================

rules['leverage'] = rules['support'] - rules['antecedent support'] * rules['consequent support']

# Para conviction evitar división por cero cuando confidence == 1
rules['conviction'] = (1 - rules['consequent support']) / (1 - rules['confidence'])
rules.loc[rules['confidence'] == 1, 'conviction'] = np.inf

# =====================
# Visualización de las métricas
# =====================

plt.figure(figsize=(18, 12))

metrics = ['support', 'confidence', 'lift', 'leverage', 'conviction']
colors = ['skyblue', 'lightgreen', 'gold', 'salmon', 'plum']

for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 3, i)
    plt.hist(rules[metric], bins=30, color=colors[i-1], edgecolor='black', alpha=0.7)
    plt.title(f'Distribución de {metric.capitalize()}')
    plt.xlabel(metric.capitalize())
    plt.ylabel('Frecuencia')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('parcial/apriori/apriori_reglas_asociacion.png', dpi=300, bbox_inches='tight')
plt.show()

# =====================
# Guardar reglas en .pkl
# =====================
with open('parcial/apriori/rules_apriori.pkl', 'wb') as f:
    pickle.dump(rules, f)

# =====================
# Visualización de las reglas
# =====================

rules = rules.sort_values('confidence', ascending=False)

print("\n=== TOP 5 REGLAS DE ASOCIACIÓN ===")
# Configurar pandas para mostrar mejor los DataFrames
with pd.option_context('display.max_colwidth', None, 'display.width', None):
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# =====================
# Monitoreo de recursos después del procesamiento
# =====================
end_time = time.time()
memory_final = process.memory_info().rss / (1024 * 1024)
peak_memory = max(peak_memory, memory_final)

execution_time = end_time - start_time
memory_increase = memory_final - memory_before

print(f"\n=== RESUMEN DE MEMORIA ===")
print(f"Memoria inicial: {memory_before:.2f} MB")
print(f"Memoria pico: {peak_memory:.2f} MB")
print(f"Memoria final: {memory_final:.2f} MB")
print(f"Incremento total: {memory_increase:.2f} MB")
print(f"Tiempo de ejecución: {execution_time:.2f} segundos")

mlflow.log_metric("execution_time_seconds", execution_time)
mlflow.log_metric("memory_before_MB", memory_before)
mlflow.log_metric("memory_peak_MB", peak_memory)
mlflow.log_metric("memory_final_MB", memory_final)
mlflow.log_metric("memory_increase_MB", memory_increase)

# =====================
# Configuración de MLflow
# =====================

with mlflow.start_run(run_name="Reglas_Apriori"):
    # Log de métricas agregadas
    mlflow.log_metric("rules_total", len(rules))
    mlflow.log_metric("avg_support", rules['support'].mean())
    mlflow.log_metric("avg_confidence", rules['confidence'].mean())
    mlflow.log_metric("avg_lift", rules['lift'].mean())
    mlflow.log_metric("max_lift", rules['lift'].max())
    mlflow.log_metric("avg_leverage", rules['leverage'].mean())
    mlflow.log_metric("avg_conviction", rules['conviction'].replace([np.inf], np.nan).dropna().mean())

    # Log del archivo con reglas
    mlflow.log_artifact('parcial/apriori/rules_apriori.pkl')

    # Log del gráfico
    mlflow.log_artifact('parcial/apriori/apriori_reglas_asociacion.png')

    # También podés loggear parámetros si usás distintos valores de soporte o confianza
    mlflow.log_param("min_support", 0.01)
    mlflow.log_param("min_confidence", 0.6)
