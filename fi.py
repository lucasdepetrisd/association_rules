import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np

# Configurar el estilo de las gráficas
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Cargar el dataset
df = pd.read_excel('online_retail_2.xlsx')

# Preprocesamiento de datos
df['Description'] = df['Description'].str.strip()
df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('str')
df = df[~df['InvoiceNo'].str.contains('C')]

# Crear matriz de transacciones (one-hot encoding)
basket = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# Convertir cantidades a booleanos (1: comprado, 0: no comprado)
basket_sets = basket.map(lambda x: True if x > 0 else False)

# Generar conjuntos frecuentes con soporte mínimo de 0.01
frequent_itemsets = apriori(basket_sets, 
                            min_support=0.01, 
                            use_colnames=True)

# Generar reglas de asociación
rules = association_rules(frequent_itemsets, 
                          metric="lift", 
                          min_threshold=1)

# Ordenar reglas por lift
rules = rules.sort_values('lift', ascending=False)

# Mostrar los primeros resultados
print("Conjuntos frecuentes más comunes:")
print(frequent_itemsets.head())
print("\nReglas de asociación más fuertes:")
print(rules.head())

# ===== VISUALIZACIONES =====

# 1. Distribución del soporte de los conjuntos frecuentes
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.hist(frequent_itemsets['support'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribución del Soporte')
plt.xlabel('Soporte')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 2. Top 10 conjuntos frecuentes por soporte
plt.subplot(2, 3, 2)
top_itemsets = frequent_itemsets.nlargest(10, 'support')
plt.barh(range(len(top_itemsets)), top_itemsets['support'], color='lightcoral')
plt.yticks(range(len(top_itemsets)), [str(items)[:30] + '...' if len(str(items)) > 30 else str(items) 
                                     for items in top_itemsets['itemsets']])
plt.title('Top 10 Conjuntos Frecuentes')
plt.xlabel('Soporte')
plt.gca().invert_yaxis()

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
top_rules = rules.head(10)
plt.barh(range(len(top_rules)), top_rules['lift'], color='mediumpurple')
plt.yticks(range(len(top_rules)), [f"{str(rule)[:25]}..." if len(str(rule)) > 25 else str(rule) 
                                  for rule in top_rules.index])
plt.title('Top 10 Reglas por Lift')
plt.xlabel('Lift')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('analisis_reglas_asociacion.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== ANÁLISIS ADICIONAL =====

# 7. Análisis de la longitud de los conjuntos frecuentes
plt.figure(figsize=(15, 10))

# Contar longitud de itemsets
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
length_counts = frequent_itemsets['length'].value_counts().sort_index()

plt.subplot(2, 3, 1)
plt.bar(length_counts.index, length_counts.values, color='coral')
plt.title('Distribución por Longitud de Conjuntos')
plt.xlabel('Número de Items')
plt.ylabel('Cantidad de Conjuntos')
plt.grid(True, alpha=0.3)

# 8. Soporte promedio por longitud
plt.subplot(2, 3, 2)
avg_support_by_length = frequent_itemsets.groupby('length')['support'].mean()
plt.bar(avg_support_by_length.index, avg_support_by_length.values, color='lightblue')
plt.title('Soporte Promedio por Longitud')
plt.xlabel('Número de Items')
plt.ylabel('Soporte Promedio')
plt.grid(True, alpha=0.3)

# 9. Heatmap de correlación entre métricas
plt.subplot(2, 3, 3)
correlation_matrix = rules[['support', 'confidence', 'lift', 'leverage', 'conviction']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Correlación entre Métricas')

# 10. Análisis de reglas por rangos de lift
plt.subplot(2, 3, 4)
lift_ranges = pd.cut(rules['lift'], bins=[0, 1, 2, 5, 10, 100], labels=['0-1', '1-2', '2-5', '5-10', '10+'])
lift_dist = lift_ranges.value_counts()
plt.pie(lift_dist.values, labels=lift_dist.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribución de Reglas por Rango de Lift')

# 11. Análisis de reglas por rangos de confianza
plt.subplot(2, 3, 5)
conf_ranges = pd.cut(rules['confidence'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                     labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'])
conf_dist = conf_ranges.value_counts()
plt.pie(conf_dist.values, labels=conf_dist.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribución de Reglas por Rango de Confianza')

# 12. Gráfico de dispersión 3D (Soporte, Confianza, Lift)
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(2, 3, 6, projection='3d')
scatter = ax.scatter(rules['support'], rules['confidence'], rules['lift'], 
                    c=rules['lift'], cmap='viridis', alpha=0.6)
ax.set_xlabel('Soporte')
ax.set_ylabel('Confianza')
ax.set_zlabel('Lift')
ax.set_title('Reglas en 3D')
plt.colorbar(scatter, ax=ax, label='Lift')

plt.tight_layout()
plt.savefig('analisis_detallado_reglas.png', dpi=300, bbox_inches='tight')
plt.show()

# ===== RESUMEN ESTADÍSTICO =====
print("\n" + "="*50)
print("RESUMEN ESTADÍSTICO DE LAS REGLAS DE ASOCIACIÓN")
print("="*50)

print(f"\nTotal de conjuntos frecuentes: {len(frequent_itemsets)}")
print(f"Total de reglas generadas: {len(rules)}")
print(f"Conjuntos de 1 item: {len(frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == 1])}")
print(f"Conjuntos de 2 items: {len(frequent_itemsets[frequent_itemsets['itemsets'].apply(len) == 2])}")
print(f"Conjuntos de 3+ items: {len(frequent_itemsets[frequent_itemsets['itemsets'].apply(len) >= 3])}")

print(f"\nEstadísticas de las reglas:")
print(f"Confianza promedio: {rules['confidence'].mean():.4f}")
print(f"Lift promedio: {rules['lift'].mean():.4f}")
print(f"Soporte promedio: {rules['support'].mean():.4f}")

print(f"\nTop 5 reglas más fuertes (por lift):")
for i, (idx, rule) in enumerate(rules.head().iterrows(), 1):
    antecedents = list(rule['antecedents'])
    consequents = list(rule['consequents'])
    print(f"{i}. {antecedents} → {consequents}")
    print(f"   Lift: {rule['lift']:.3f}, Confianza: {rule['confidence']:.3f}, Soporte: {rule['support']:.3f}")
