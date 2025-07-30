import pandas as pd

# ======================
# Cargar modelo .pkl
# ======================

def load_model(file_path):
    try:
        model = pd.read_pickle(file_path)
        print(f"Modelo cargado desde {file_path}")
        return model
    except FileNotFoundError:
        print(f"El archivo {file_path} no se encontró.")
        return None

# =====================
# Función de recomendación para productos
# =====================

def get_product_recommendations(product_name, rules_df):
    matched_rules = rules_df[rules_df['antecedents'].apply(
        lambda x: product_name in x)]

    if matched_rules.empty:
        print(f"No se encontraron reglas para el producto '{product_name}'")
        return None

    # Mostrar los datos de forma más legible
    print(f"\nRecomendaciones para quienes compraron '{product_name}':")
    print("=" * 80)

    for idx, row in matched_rules.iterrows():
        print(f"\nRegla {idx}:")
        print(f"  Antecedentes: {list(row['antecedents'])}")
        print(f"  Consecuentes: {list(row['consequents'])}")
        print(f"  Confianza: {row['confidence']:.3f}")
        if 'lift' in row:
            print(f"  Lift: {row['lift']:.3f}")
        if 'support' in row:
            print(f"  Support: {row['support']:.3f}")
        print("-" * 40)

    # Configurar pandas para mostrar mejor los DataFrames
    with pd.option_context('display.max_colwidth', None, 'display.width', None):
        print("\nTabla resumida de recomendaciones:")
        print(matched_rules[['antecedents', 'consequents',
              'confidence', 'lift', 'support']].head(10))

    return matched_rules[['antecedents', 'consequents', 'confidence']].head()


rules = load_model('parcial/apriori/rules_apriori.pkl')

# Ejemplo de uso

producto = 'WOODEN TREE CHRISTMAS SCANDINAVIAN'  # Cambiar por un producto válido
recommendations = get_product_recommendations(producto, rules)
