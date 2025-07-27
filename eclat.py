import pandas as pd
import numpy as np
from collections import defaultdict

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

# Función para implementar ECLAT


def eclat(transactions, min_support=0.2):
    # Crear diccionario de transacciones verticales (TID-list)
    tid_lists = defaultdict(set)
    for tid, items in enumerate(transactions):
        for item in items:
            tid_lists[item].add(tid)

    # Filtrar items por soporte mínimo
    n_transactions = len(transactions)
    min_support_count = min_support * n_transactions

    frequent_items = {item: tids for item, tids in tid_lists.items()
                      if len(tids) >= min_support_count}

    # Generar conjuntos frecuentes de tamaño k
    def generate_frequent_itemsets(prefix, tid_list, k):
        frequent_itemsets = []
        if k > 0:
            items = list(frequent_items.keys())
            for _, item in enumerate(items):
                new_prefix = prefix + [item]
                new_tid_list = tid_list & frequent_items[item]
                if len(new_tid_list) >= min_support_count:
                    frequent_itemsets.append((new_prefix, new_tid_list))
                    frequent_itemsets.extend(
                        generate_frequent_itemsets(
                            new_prefix, new_tid_list, k-1)
                    )
        return frequent_itemsets

    # Encontrar todos los conjuntos frecuentes
    all_frequent_itemsets = []
    for item in frequent_items:
        prefix = [item]
        tid_list = frequent_items[item]
        all_frequent_itemsets.append((prefix, tid_list))
        all_frequent_itemsets.extend(
            generate_frequent_itemsets(prefix, tid_list, 2)
        )

    # Formatear resultados
    results = []
    for itemset, tids in all_frequent_itemsets:
        support = len(tids) / n_transactions
        results.append({
            'itemset': itemset,
            'support': support
        })

    return pd.DataFrame(results)


# Ejecutar el algoritmo ECLAT
df = pd.DataFrame(data)
frequent_itemsets = eclat(df['Películas'], min_support=0.2)

# Ordenar por soporte
frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)

print("Conjuntos frecuentes encontrados:")
print(frequent_itemsets)
