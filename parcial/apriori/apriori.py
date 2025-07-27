import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from mlxtend.frequent_patterns import apriori, association_rules
from efficient_apriori import apriori as efficient_apriori_alg

# =====================
# Cargar el dataset
# =====================
df = pd.read_excel('online_retail_2.xlsx')

# =====================
# Inspección Inicial y Limpieza Básica
# =====================
print("Información general del DataFrame:")
df.info()

missing_customers = df['CustomerID'].isnull().sum()
total_rows = df.shape[0]
print(f"\nSe encontraron {missing_customers} filas sin CustomerID ({missing_customers/total_rows:.2%}).")

df.dropna(subset=['CustomerID'], inplace=True)
print(f"Filas después de eliminar CustomerID nulos: {df.shape[0]}")

df['CustomerID'] = df['CustomerID'].astype(int)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True)

# =====================
# Limpieza de Registros Inválidos
# =====================
print(f"Filas con cantidad negativa (devoluciones): {df[df['Quantity'] <= 0].shape[0]}")
df = df[df['Quantity'] > 0]
print(f"Filas después de eliminar cantidades negativas: {df.shape[0]}")

print(f"\nFilas con precio unitario cero: {df[df['UnitPrice'] <= 0].shape[0]}")
df = df[df['UnitPrice'] > 0]
print(f"Filas después de eliminar precios cero: {df.shape[0]}")

# =====================
# Foco en el Mercado Principal (Reino Unido)
# =====================
print("Distribución de clientes por país (Top 10):")
print(df['Country'].value_counts().head(10))

df_uk = df[df['Country'] == 'United Kingdom'].copy()
print(f"\nAnálisis enfocado en el Reino Unido. Total de filas: {df_uk.shape[0]}")

# =====================
# Eliminación de Duplicados
# =====================
print(f"Número de filas duplicadas: {df_uk.duplicated().sum()}")
df_uk.drop_duplicates(inplace=True)
print(f"Filas después de eliminar duplicados: {df_uk.shape[0]}")

# =====================
# Selección de Productos Más Comprados
# =====================
top_products = df_uk['Description'].value_counts().head(500).index
df_uk = df_uk[df_uk['Description'].isin(top_products)]
print("Productos más comprados:")
print(df_uk['Description'].value_counts().head(10))

# =====================
# Cantidad de filas
# =====================

print(f"Total de filas después de filtrar productos: {df_uk.shape[0]}")

# =====================
# Preprocesamiento adicional para Apriori
# =====================
df_uk['Description'] = df_uk['Description'].str.strip()
df_uk.dropna(subset=['InvoiceNo'], inplace=True)
df_uk['InvoiceNo'] = df_uk['InvoiceNo'].astype(str)
df_uk = df_uk[~df_uk['InvoiceNo'].str.contains('C')]  # Quitar devoluciones
df_uk = df_uk[df_uk['Quantity'] > 0]

# =====================
# Crear matriz de transacciones
# =====================
basket = (df_uk.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

# =====================
# Apriori y reglas de asociación
# =====================
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.6)

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
plt.savefig('apriori_reglas_asociacion.png', dpi=300, bbox_inches='tight')
plt.show()

# =====================
# Guardar reglas en .pkl
# =====================
with open('rules_apriori.pkl', 'wb') as f:
    pickle.dump(rules, f)

# =====================
# Visualización de las reglas
# =====================

rules = rules.sort_values('confidence', ascending=False)

print("Top 5 reglas de asociación:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']].head())

# =====================
# Función de recomendación para productos
# =====================

def get_product_recommendations(product_name):
    matched_rules = rules[rules['antecedents'].apply(lambda x: product_name in x)]
    return matched_rules[['antecedents', 'consequents', 'confidence']].head()

# Ejemplo de uso
producto = 'WHITE HANGING HEART T-LIGHT HOLDER'  # Cambiar por un producto válido
print(f"\nRecomendaciones para quienes compraron '{producto}':")
print(get_product_recommendations(producto))
