import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Crear un dataset de ejemplo de películas vistas por usuarios
data = {
    'UserID': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5],
    'Película': ['Star Wars', 'Matrix', 'El Señor de los Anillos', 
                 'Matrix', 'Inception', 
                 'El Señor de los Anillos', 'Star Wars', 'Inception',
                 'Matrix', 'Interstellar',
                 'Star Wars', 'Inception', 'Interstellar']
}

# Crear DataFrame
df = pd.DataFrame(data)

# Convertir los datos a formato one-hot encoding
movies_encoded = pd.get_dummies(df['Película']).groupby(df['UserID']).max()

# Aplicar el algoritmo Apriori para encontrar conjuntos frecuentes de películas
frequent_itemsets = apriori(movies_encoded, min_support=0.2, use_colnames=True)

# Generar reglas de asociación
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Ordenar reglas por confianza
rules = rules.sort_values('confidence', ascending=False)

# Mostrar las top 5 recomendaciones
print("Top 5 recomendaciones de películas:")
print(rules[['antecedents', 'consequents', 'support', 'confidence']].head())

# Función para obtener recomendaciones para una película específica
def get_movie_recommendations(movie_name):
    movie_recs = rules[rules['antecedents'].apply(lambda x: movie_name in x)]
    return movie_recs[['antecedents', 'consequents', 'confidence']].head()

# Ejemplo de uso
print("\nRecomendaciones para fans de 'Matrix':")
print(get_movie_recommendations('Matrix'))
