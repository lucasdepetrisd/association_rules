import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import mlflow
from itertools import combinations

from resource_monitor import ResourceMonitor, get_dataset_memory_usage, print_initial_system_info
from output_capture import setup_output_capture, finalize_output_capture, print_script_header, print_script_footer

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Reglas_ECLAT")

# =====================
# Capturar salida de print
# =====================

# Iniciar captura de salida
output_capture = setup_output_capture()

print_script_header("ECLAT", "Algoritmo ECLAT")

# =====================
# Cargar el dataset
# =====================
df = pd.read_csv('cleaned_online_retail.csv')

# =====================
# Preprocesamiento adicional para ECLAT
# =====================
df['Description'] = df['Description'].str.strip()
df.dropna(subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype(str)
df = df[~df['InvoiceNo'].str.contains('C')]  # Quitar devoluciones
df = df[df['Quantity'] > 0]

# =====================
# Crear matriz de transacciones para ECLAT
# =====================
basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

# =====================
# Implementación manual de ECLAT
# =====================

class ECLATImplementation:
    """Implementación manual del algoritmo ECLAT"""

    def __init__(self, data, min_support=0.01):
        self.data = data
        self.min_support = min_support
        self.total_transactions = len(data)
        self.min_support_count = int(min_support * self.total_transactions)
        self.frequent_itemsets = {}
        self.item_tidsets = {}

    def get_item_tidsets(self):
        """Obtener TID-sets para cada item individual"""
        for item in self.data.columns:
            tidset = set(self.data[self.data[item] == 1].index)
            if len(tidset) >= self.min_support_count:
                self.item_tidsets[frozenset([item])] = tidset
                self.frequent_itemsets[frozenset([item])] = len(tidset) / self.total_transactions

        print(f"Items frecuentes (1-itemsets): {len(self.item_tidsets)}")

    def eclat_recursive(self, itemsets, k):
        """Algoritmo ECLAT recursivo"""
        if not itemsets:
            return

        next_itemsets = {}
        itemsets_list = list(itemsets.keys())

        for i in range(len(itemsets_list)):
            for j in range(i + 1, len(itemsets_list)):
                itemset1 = itemsets_list[i]
                itemset2 = itemsets_list[j]

                # Verificar si se pueden combinar (prefijo común)
                union_itemset = itemset1.union(itemset2)
                if len(union_itemset) == k:
                    # Intersección de TID-sets
                    tidset_intersection = itemsets[itemset1].intersection(itemsets[itemset2])

                    if len(tidset_intersection) >= self.min_support_count:
                        next_itemsets[union_itemset] = tidset_intersection
                        support = len(tidset_intersection) / self.total_transactions
                        self.frequent_itemsets[union_itemset] = support

        if next_itemsets:
            print(f"Itemsets frecuentes de tamaño {k}: {len(next_itemsets)}")
            self.eclat_recursive(next_itemsets, k + 1)

    def fit(self):
        """Ejecutar el algoritmo ECLAT"""
        print("\n=== INICIANDO ALGORITMO ECLAT ===")
        print("Calculando 1-itemsets frecuentes...")
        self.get_item_tidsets()

        print("Generando itemsets de mayor tamaño...")
        self.eclat_recursive(self.item_tidsets, 2)

        print(f"Total de itemsets frecuentes encontrados: {len(self.frequent_itemsets)}")
        return self.frequent_itemsets

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
# Ejecutar ECLAT
# =====================

# Crear instancia de ECLAT
eclat_algo = ECLATImplementation(basket_sets, min_support=0.01)

# Ejecutar algoritmo
frequent_itemsets = eclat_algo.fit()

# =====================
# Generar reglas de asociación desde itemsets frecuentes
# =====================

print("\nGenerando reglas de asociación...")

def generate_association_rules(frequent_itemsets, min_confidence=0.6):
    """Generar reglas de asociación desde itemsets frecuentes"""
    rules = []
    
    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:  # Skip 1-itemsets
            continue
            
        items = list(itemset)
        
        # Generar todas las posibles reglas A -> B
        for i in range(1, len(items)):
            for antecedent_items in combinations(items, i):
                antecedent = frozenset(antecedent_items)
                consequent = itemset - antecedent
                
                if antecedent in frequent_itemsets and consequent in frequent_itemsets:
                    support_antecedent = frequent_itemsets[antecedent]
                    support_consequent = frequent_itemsets[consequent]
                    
                    confidence = support / support_antecedent
                    
                    if confidence >= min_confidence:
                        lift = confidence / support_consequent if support_consequent > 0 else 0
                        leverage = support - support_antecedent * support_consequent
                        
                        # Para conviction evitar división por cero cuando confidence == 1
                        conviction = (1 - support_consequent) / (1 - confidence) if confidence < 1 else np.inf
                        
                        rules.append({
                            'antecedents': antecedent,
                            'consequents': consequent,
                            'antecedent support': support_antecedent,
                            'consequent support': support_consequent,
                            'support': support,
                            'confidence': confidence,
                            'lift': lift,
                            'leverage': leverage,
                            'conviction': conviction
                        })
    
    return pd.DataFrame(rules)

rules = generate_association_rules(frequent_itemsets, min_confidence=0.6)
print(f"Reglas generadas: {len(rules)}")

# =====================
# Visualización de las métricas
# =====================

plt.figure(figsize=(18, 12))

# 1. Distribución del soporte
plt.subplot(2, 3, 1)
plt.hist(rules['support'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribución del Soporte')
plt.xlabel('Soporte')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 2. Distribución de la confianza
plt.subplot(2, 3, 2)
plt.hist(rules['confidence'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Distribución de la Confianza')
plt.xlabel('Confianza')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 3. Distribución del lift
plt.subplot(2, 3, 3)
plt.hist(rules['lift'], bins=30, alpha=0.7, color='gold', edgecolor='black')
plt.title('Distribución del Lift')
plt.xlabel('Lift')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 4. Distribución del leverage
plt.subplot(2, 3, 4)
plt.hist(rules['leverage'], bins=30, alpha=0.7, color='salmon', edgecolor='black')
plt.title('Distribución del Leverage')
plt.xlabel('Leverage')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 5. Scatter plot: Confianza vs Soporte (coloreado por Lift)
plt.subplot(2, 3, 5)
scatter = plt.scatter(rules['support'], rules['confidence'], alpha=0.6, c=rules['lift'], cmap='viridis')
plt.colorbar(scatter, label='Lift')
plt.title('Confianza vs Soporte (coloreado por Lift)')
plt.xlabel('Soporte')
plt.ylabel('Confianza')
plt.grid(True, alpha=0.3)

# 6. Top 10 reglas por lift
plt.subplot(2, 3, 6)
if len(rules) > 0:
    top_rules = rules.nlargest(10, 'lift')
    y_pos = range(len(top_rules))
    plt.barh(y_pos, top_rules['lift'], color='mediumpurple')
    # Crear etiquetas más cortas para las reglas
    rule_labels = []
    for idx in top_rules.index:
        antecedents = str(list(top_rules.loc[idx, 'antecedents']))[:20]
        consequents = str(list(top_rules.loc[idx, 'consequents']))[:20]
        rule_labels.append(f"{antecedents}→{consequents}")
    plt.yticks(y_pos, rule_labels, fontsize=8)
    plt.title('Top 10 Reglas por Lift')
    plt.xlabel('Lift')
    plt.gca().invert_yaxis()
else:
    plt.text(0.5, 0.5, 'No hay reglas para mostrar', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Top 10 Reglas por Lift')

plt.tight_layout()
plt.savefig('parcial/eclat/eclat_reglas_asociacion.png', dpi=300, bbox_inches='tight')
plt.show()

# =====================
# Gráfico adicional de distribución de conviction
# =====================

plt.figure(figsize=(12, 8))

# Subplot 1: Distribución de conviction (sin infinitos)
plt.subplot(2, 2, 1)
if len(rules) > 0:
    conviction_finite = rules['conviction'].replace([np.inf], np.nan).dropna()
    if len(conviction_finite) > 0:
        plt.hist(conviction_finite, bins=30, alpha=0.7, color='plum', edgecolor='black')
    else:
        plt.text(0.5, 0.5, 'No hay datos finitos de conviction', ha='center', va='center', transform=plt.gca().transAxes)
else:
    plt.text(0.5, 0.5, 'No hay reglas', ha='center', va='center', transform=plt.gca().transAxes)
plt.title('Distribución de Conviction (sin inf)')
plt.xlabel('Conviction')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# Subplot 2: Scatter plot Lift vs Confidence
plt.subplot(2, 2, 2)
if len(rules) > 0:
    plt.scatter(rules['confidence'], rules['lift'], alpha=0.6, color='orange')
else:
    plt.text(0.5, 0.5, 'No hay reglas para mostrar', ha='center', va='center', transform=plt.gca().transAxes)
plt.title('Lift vs Confianza')
plt.xlabel('Confianza')
plt.ylabel('Lift')
plt.grid(True, alpha=0.3)

# Subplot 3: Distribución del número de items en antecedentes
plt.subplot(2, 2, 3)
if len(rules) > 0:
    antecedent_lengths = rules['antecedents'].apply(len)
    if len(antecedent_lengths) > 0:
        plt.hist(antecedent_lengths, bins=range(1, max(antecedent_lengths)+2), alpha=0.7, color='cyan', edgecolor='black')
    else:
        plt.text(0.5, 0.5, 'No hay datos de antecedentes', ha='center', va='center', transform=plt.gca().transAxes)
else:
    plt.text(0.5, 0.5, 'No hay reglas', ha='center', va='center', transform=plt.gca().transAxes)
plt.title('Distribución del Tamaño de Antecedentes')
plt.xlabel('Número de Items en Antecedentes')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# Subplot 4: Estadísticas resumen
plt.subplot(2, 2, 4)
plt.axis('off')
if len(rules) > 0:
    stats_text = f"""
Resumen Estadístico ECLAT:

Número total de reglas: {len(rules)}
Número de frequent itemsets: {len(frequent_itemsets)}

Métricas promedio:
• Soporte: {rules['support'].mean():.4f}
• Confianza: {rules['confidence'].mean():.4f}
• Lift: {rules['lift'].mean():.4f}
• Leverage: {rules['leverage'].mean():.4f}

Métricas máximas:
• Lift máximo: {rules['lift'].max():.4f}
• Confianza máxima: {rules['confidence'].max():.4f}
"""
else:
    stats_text = """
Resumen Estadístico ECLAT:

No se generaron reglas de asociación
con los parámetros especificados.

Intenta reducir min_confidence o 
min_support para obtener más reglas.
"""

plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('parcial/eclat/eclat_analisis_detallado.png', dpi=300, bbox_inches='tight')
plt.show()

# =====================
# Guardar reglas en .pkl
# =====================
with open('parcial/eclat/rules_eclat.pkl', 'wb') as f:
    pickle.dump(rules, f)

# =====================
# Visualización de las reglas
# =====================

if len(rules) > 0:
    rules = rules.sort_values('confidence', ascending=False)

    print("\n=== TOP 5 REGLAS DE ASOCIACIÓN ===")
    # Configurar pandas para mostrar mejor los DataFrames
    with pd.option_context('display.max_colwidth', None, 'display.width', None):
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
else:
    print("\n=== NO SE ENCONTRARON REGLAS DE ASOCIACIÓN ===")
    print("Considera reducir min_confidence o min_support")

# =====================
# Monitoreo de recursos después del procesamiento
# =====================

# Detener monitoreo y obtener métricas
metrics = monitor.stop_monitoring(clean_memory=True)

# Imprimir resumen completo
monitor.print_summary(metrics)

# Mostrar progresión de memoria y CPU
monitor.print_progression_sample(sample_every=10)

print_script_footer("ECLAT")

# =====================
# Guardar salida capturada
# =====================

# Finalizar captura y guardar
output_filename = 'parcial/eclat/eclat_execution_log.txt'
captured_output = finalize_output_capture(output_capture, output_filename)

# =====================
# Configuración de MLflow
# =====================

with mlflow.start_run(run_name="Reglas_ECLAT"):
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
    mlflow.log_metric("itemsets_total", len(frequent_itemsets))
    mlflow.log_metric("rules_total", len(rules))

    if len(rules) > 0:
        mlflow.log_metric("avg_support", rules['support'].mean())
        mlflow.log_metric("avg_confidence", rules['confidence'].mean())
        mlflow.log_metric("avg_lift", rules['lift'].mean())
        mlflow.log_metric("max_lift", rules['lift'].max())
        mlflow.log_metric("avg_leverage", rules['leverage'].mean())
        mlflow.log_metric("avg_conviction", rules['conviction'].replace([np.inf], np.nan).dropna().mean())
    else:
        # Si no hay reglas, loggear métricas en 0
        for metric in ['avg_support', 'avg_confidence', 'avg_lift', 'max_lift', 'avg_leverage', 'avg_conviction']:
            mlflow.log_metric(metric, 0.0)

    # Log del archivo con reglas
    mlflow.log_artifact('parcial/eclat/rules_eclat.pkl')

    # Log de los gráficos
    mlflow.log_artifact('parcial/eclat/eclat_reglas_asociacion.png')
    mlflow.log_artifact('parcial/eclat/eclat_analisis_detallado.png')

    # Log de la salida completa de ejecución
    mlflow.log_artifact(output_filename)

    # También podés loggear parámetros si usás distintos valores de soporte o confianza
    mlflow.log_param("min_support", 0.01)
    mlflow.log_param("min_confidence", 0.6)

print("Ejecución completada. Todos los artifacts guardados en MLflow.")
