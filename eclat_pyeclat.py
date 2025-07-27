from pyECLAT import ECLAT
import pandas as pd

# Crear un dataset de ejemplo de transacciones de películas
data = {
    'TransaccionID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Películas': [
        ['Star Wars', 'Matrix', 'El Señor de los Anillos'],
        ['Matrix', 'Inception'],
        ['El Señor de los Anillos', 'Star Wars', 'Inception'],
        ['Matrix', 'Interstellar'], 
        ['Star Wars', 'Inception', 'Interstellar'],
        ['Matrix', 'El Señor de los Anillos', 'Interstellar'],
        ['Star Wars', 'Matrix', 'Inception', 'Interstellar'],
        ['El Señor de los Anillos', 'Inception'],
        ['Star Wars', 'El Señor de los Anillos'],
        ['Matrix', 'Star Wars', 'Inception', 'El Señor de los Anillos']
    ]
}

# Crear DataFrame
df = pd.DataFrame(data)

# Preparar datos para ECLAT - convertir a formato de matriz binaria
transactions_list = df['Películas'].tolist()

# Obtener todos los items únicos
all_items = set()
for transaction in transactions_list:
    all_items.update(transaction)

# Crear matriz binaria
binary_matrix = []
for transaction in transactions_list:
    row = []
    for item in sorted(all_items):
        row.append(1 if item in transaction else 0)
    binary_matrix.append(row)

# Crear DataFrame con formato correcto para pyECLAT
eclat_df = pd.DataFrame(binary_matrix, columns=sorted(all_items))

# Crear instancia de ECLAT
eclat_instance = ECLAT(eclat_df)

# Ejecutar ECLAT con soporte mínimo de 0.2 (20%)
# method=0 para obtener solo conjuntos frecuentes
frequent_itemsets = eclat_instance.fit(min_support=0.2, method=0)

print("Conjuntos frecuentes encontrados con pyECLAT:")
print(frequent_itemsets)

# Obtener reglas de asociación
# method=1 para obtener reglas de asociación
rules = eclat_instance.fit(min_support=0.2, min_confidence=0.6, method=1)

print("\nReglas de asociación encontradas:")
print(rules)
