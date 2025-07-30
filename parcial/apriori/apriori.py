import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from mlxtend.frequent_patterns import apriori, association_rules
import mlflow
import sys
import io
from resource_monitor import ResourceMonitor, get_dataset_memory_usage, print_initial_system_info

# Configurar pandas para mostrar mejor los datos
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Reglas_Apriori")

# =====================
# Capturar salida de print
# =====================

class TeeOutput:
    """Clase para capturar print y mantener salida en consola"""
    def __init__(self):
        self.terminal = sys.stdout
        self.log = io.StringIO()
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def get_output(self):
        return self.log.getvalue()

# Iniciar captura de salida
output_capture = TeeOutput()
sys.stdout = output_capture

print("=== INICIO DE EJECUCIÓN APRIORI ===")
print(f"Timestamp: {pd.Timestamp.now()}")
print("="*50)

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

# Crear instancia del monitor
monitor = ResourceMonitor(monitoring_interval=0.5)

# Iniciar monitoreo
initial_info = monitor.start_monitoring(clean_memory=True)

# Mostrar información inicial del sistema
print_initial_system_info(initial_info)
get_dataset_memory_usage(df, "dataset original")
get_dataset_memory_usage(basket_sets, "basket_sets")

# =====================
# Apriori y reglas de asociación
# =====================
print("\n=== INICIANDO ALGORITMO APRIORI ===")

# Medir memoria durante frequent_itemsets
print("Calculando frequent itemsets...")
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
print(f"Frequent itemsets encontrados: {len(frequent_itemsets)}")

# Medir memoria durante association_rules
print("Generando reglas de asociación...")
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)
print(f"Reglas generadas: {len(rules)}")

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

# Detener monitoreo y obtener métricas
metrics = monitor.stop_monitoring(clean_memory=True)

# Imprimir resumen completo
monitor.print_summary(metrics)

# Mostrar progresión de memoria y CPU
monitor.print_progression_sample(sample_every=10)

print(f"\n=== FIN DE EJECUCIÓN APRIORI ===")
print(f"Timestamp: {pd.Timestamp.now()}")
print("="*50)

# =====================
# Guardar salida capturada
# =====================

# Restaurar stdout original
sys.stdout = output_capture.terminal

# Obtener la salida capturada
captured_output = output_capture.get_output()

# Guardar en archivo
output_filename = 'parcial/apriori/apriori_execution_log.txt'
with open(output_filename, 'w', encoding='utf-8') as f:
    f.write(captured_output)

# =====================
# Configuración de MLflow
# =====================

with mlflow.start_run(run_name="Reglas_Apriori"):
    # Métricas de rendimiento
    mlflow.log_metric("execution_time_seconds", metrics['execution_time_seconds'])

    # Métricas de memoria
    mlflow.log_metric("memory_initial_MB", metrics['memory_initial_MB'])
    mlflow.log_metric("memory_final_MB", metrics['memory_final_MB'])
    mlflow.log_metric("memory_peak_process_MB", metrics['memory_peak_process_MB'])
    mlflow.log_metric("memory_peak_system_MB", metrics['memory_peak_system_MB'])
    mlflow.log_metric("memory_max_additional_MB", metrics['memory_max_additional_MB'])
    mlflow.log_metric("memory_freed_after_peak_MB", metrics['memory_freed_after_peak_MB'])
    mlflow.log_metric("max_system_percent", metrics['max_system_percent'])

    # Métricas de CPU
    mlflow.log_metric("cpu_peak_process_percent", metrics['cpu_peak_process_percent'])
    mlflow.log_metric("cpu_peak_system_percent", metrics['cpu_peak_system_percent'])
    mlflow.log_metric("cpu_avg_process_percent", metrics['cpu_avg_process_percent'])
    mlflow.log_metric("cpu_avg_system_percent", metrics['cpu_avg_system_percent'])
    mlflow.log_metric("cpu_time_used_seconds", metrics['cpu_time_used_seconds'])
    mlflow.log_metric("cpu_efficiency_percent", metrics['cpu_efficiency_percent'])

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
    
    # Log de la salida completa de ejecución
    mlflow.log_artifact(output_filename)

    # También podés loggear parámetros si usás distintos valores de soporte o confianza
    mlflow.log_param("min_support", 0.01)
    mlflow.log_param("min_confidence", 0.6)

print("Ejecución completada. Todos los artifacts guardados en MLflow.")
