from pyECLAT import ECLAT
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Dataset de ejemplo
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
df = pd.DataFrame(data)

# Expandir las listas en columnas (formato transaccional esperado por pyECLAT)
df_expanded = df['Películas'].apply(pd.Series)
df_expanded = df_expanded.reset_index(drop=True)
df_expanded.columns = range(df_expanded.shape[1])

# Crear instancia y obtener combinaciones frecuentes (solo combos 2 y 3)
eclat = ECLAT(data=df_expanded, verbose=True)
indexes = eclat.fit(
    min_support=0.2,
    min_combination=2,
    max_combination=3,
    separator=' & ',
    verbose=True
)
index_dict = indexes[0]

# Obtener soporte de todos los itemsets
supports = eclat.support()

# Construir DataFrame de reglas a partir de combinaciones frecuentes (pares y tríos)
rules_list = []
total_transactions = df.shape[0]

for combo_str, tids in index_dict.items():
    items = combo_str.split(' & ')
    tids_set = set(tids)
    support = supports[combo_str]

    # Para cada regla posible A->C donde A y C son subconjuntos disjuntos del itemset
    # En combos de tamaño 2 o 3
    itemset = set(items)
    for r in range(1, len(items)):
        for antecedent in combinations(items, r):
            antecedent = set(antecedent)
            consequent = itemset - antecedent

            # calcular soporte de antecedent y consequent
            antecedent_str = ' & '.join(sorted(antecedent))
            consequent_str = ' & '.join(sorted(consequent))

            support_antecedent = supports.get(antecedent_str, 0)
            support_consequent = supports.get(consequent_str, 0)
            confidence = support / support_antecedent if support_antecedent > 0 else 0
            lift = confidence / support_consequent if support_consequent > 0 else 0
            leverage = support - support_antecedent * support_consequent

            rules_list.append({
                'antecedents': antecedent_str,
                'consequents': consequent_str,
                'support': support,
                'confidence': confidence,
                'lift': lift,
                'leverage': leverage
            })

rules_df = pd.DataFrame(rules_list)

# Visualización
plt.figure(figsize=(15, 10))

# 1. Distribución del soporte
plt.subplot(2, 3, 1)
plt.hist(rules_df['support'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribución del Soporte')
plt.xlabel('Soporte')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 2. Distribución del leverage
plt.subplot(2, 3, 2)
plt.hist(rules_df['leverage'], bins=30, alpha=0.7, color='salmon', edgecolor='black')
plt.title('Distribución del Leverage')
plt.xlabel('Leverage')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 3. Distribución de la confianza
plt.subplot(2, 3, 3)
plt.hist(rules_df['confidence'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Distribución de la Confianza')
plt.xlabel('Confianza')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 4. Distribución del lift
plt.subplot(2, 3, 4)
plt.hist(rules_df['lift'], bins=30, alpha=0.7, color='gold', edgecolor='black')
plt.title('Distribución del Lift')
plt.xlabel('Lift')
plt.ylabel('Frecuencia')
plt.grid(True, alpha=0.3)

# 5. Scatter plot: Confianza vs Soporte coloreado por Lift
plt.subplot(2, 3, 5)
scatter = plt.scatter(rules_df['support'], rules_df['confidence'], alpha=0.6, c=rules_df['lift'], cmap='viridis')
plt.colorbar(scatter, label='Lift')
plt.title('Confianza vs Soporte (coloreado por Lift)')
plt.xlabel('Soporte')
plt.ylabel('Confianza')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizacion_reglas_eclat.png', dpi=300, bbox_inches='tight')
plt.show()
