import pandas as pd

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
# Guardar el DataFrame limpio
# =====================

df_uk.to_csv('cleaned_online_retail.csv', index=False)
