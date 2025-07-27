import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar y preparar los datos
df = pd.read_excel('online_retail_2.xlsx')

# Limpieza básica de datos
df = df.dropna()
df = df[df['Quantity'] > 0]
df = df[df['Price'] > 0]

# Crear la matriz de transacciones
basket = df.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack().fillna(0)
basket_sets = (basket > 0).astype(int)

# Aplicar el algoritmo FP-Growth
frequent_itemsets = fpgrowth(basket_sets, min_support=0.01, use_colnames=True)

# Generar reglas de asociación
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Calcular métricas adicionales
rules['leverage'] = rules.apply(lambda x: 
    x['support'] - (x['support_antecedent'] * x['support_consequent']), axis=1)
rules['conviction'] = rules.apply(lambda x: 
    (1 - x['support_consequent']) / (1 - x['confidence']) if x['confidence'] < 1 else np.inf, axis=1)

# Visualización de resultados
plt.figure(figsize=(15, 10))

# 1. Distribución del soporte
plt.subplot(2, 3, 1)
plt.hist(rules['support'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribución del Soporte')
plt.xlabel('Soporte')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 2. Distribución del leverage
plt.subplot(2, 3, 2)
plt.hist(rules['leverage'], bins=30, alpha=0.7, color='salmon', edgecolor='black')
plt.title('Distribución del Leverage')
plt.xlabel('Leverage')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 3. Distribución de la confianza
plt.subplot(2, 3, 3)
plt.hist(rules['confidence'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Distribución de la Confianza')
plt.xlabel('Confianza')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 4. Distribución del lift
plt.subplot(2, 3, 4)
plt.hist(rules['lift'], bins=30, alpha=0.7, color='gold', edgecolor='black')
plt.title('Distribución del Lift')
plt.xlabel('Lift')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 5. Scatter plot: Confianza vs Soporte
plt.subplot(2, 3, 5)
plt.scatter(rules['support'], rules['confidence'], alpha=0.6, c=rules['lift'], cmap='viridis')
plt.colorbar(label='Lift')
plt.title('Confianza vs Soporte (coloreado por Lift)')
plt.xlabel('Soporte')
plt.ylabel('Confianza')
plt.grid(True, alpha=0.3)

# 6. Top 10 reglas por lift
plt.subplot(2, 3, 6)
top_rules = rules.nlargest(10, 'lift')
plt.barh(range(len(top_rules)), top_rules['lift'], color='mediumpurple')
plt.yticks(range(len(top_rules)), [f"{str(rule)[:25]}..." if len(str(rule)) > 25 else str(rule) 
                                  for rule in top_rules.index])
plt.title('Top 10 Reglas por Lift')
plt.xlabel('Lift')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('analisis_reglas_asociacion.png', dpi=300, bbox_inches='tight')
plt.show()

# Imprimir resumen estadístico
print("\nResumen Estadístico de las Reglas de Asociación:")
print("\nNúmero total de reglas:", len(rules))
print("\nEstadísticas de las métricas principales:")
print(rules[['support', 'confidence', 'lift']].describe())

# Guardar resultados
rules.to_csv('reglas_asociacion.csv', index=False)
frequent_itemsets.to_csv('conjuntos_frecuentes.csv', index=False)
