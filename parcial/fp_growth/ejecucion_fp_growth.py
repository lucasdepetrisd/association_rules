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
        if 'leverage' in row:
            print(f"  Leverage: {row['leverage']:.3f}")
        if 'conviction' in row:
            print(f"  Conviction: {row['conviction']:.3f}")
        print("-" * 40)

    # Configurar pandas para mostrar mejor los DataFrames
    with pd.option_context('display.max_colwidth', None, 'display.width', None):
        print("\nTabla resumida de recomendaciones:")
        columns_to_show = ['antecedents', 'consequents',
                           'confidence', 'lift', 'support']
        if 'leverage' in matched_rules.columns:
            columns_to_show.append('leverage')
        if 'conviction' in matched_rules.columns:
            columns_to_show.append('conviction')
        print(matched_rules[columns_to_show].head(10))

    return matched_rules[['antecedents', 'consequents', 'confidence', 'lift']].head()

# =====================
# Función para mostrar estadísticas generales
# =====================


def show_general_stats(rules_df):
    print("\n=== ESTADÍSTICAS GENERALES FP-GROWTH ===")
    print("=" * 50)
    print(f"Número total de reglas: {len(rules_df)}")
    print(f"Soporte promedio: {rules_df['support'].mean():.4f}")
    print(f"Confianza promedio: {rules_df['confidence'].mean():.4f}")
    print(f"Lift promedio: {rules_df['lift'].mean():.4f}")
    print(f"Lift máximo: {rules_df['lift'].max():.4f}")

    if 'leverage' in rules_df.columns:
        print(f"Leverage promedio: {rules_df['leverage'].mean():.4f}")

    if 'conviction' in rules_df.columns:
        conviction_finite = rules_df['conviction'].replace(
            [float('inf')], float('nan')).dropna()
        if len(conviction_finite) > 0:
            print(
                f"Conviction promedio (sin inf): {conviction_finite.mean():.4f}")

    print("\n=== TOP 5 REGLAS POR LIFT ===")
    top_rules = rules_df.nlargest(5, 'lift')
    with pd.option_context('display.max_colwidth', None, 'display.width', None):
        columns_to_show = ['antecedents', 'consequents',
                           'support', 'confidence', 'lift']
        if 'leverage' in rules_df.columns:
            columns_to_show.append('leverage')
        print(top_rules[columns_to_show])

# =====================
# Función para buscar productos disponibles
# =====================


def search_products(rules_df, search_term):
    """Busca productos que contengan el término de búsqueda"""
    all_products = set()

    # Extraer todos los productos de antecedentes y consecuentes
    for rule_idx, rule in rules_df.iterrows():
        all_products.update(rule['antecedents'])
        all_products.update(rule['consequents'])

    # Buscar productos que contengan el término
    matching_products = [
        prod for prod in all_products if search_term.lower() in prod.lower()]

    print(f"\nProductos que contienen '{search_term}':")
    print("-" * 40)
    for i, product in enumerate(sorted(matching_products), 1):
        print(f"{i}. {product}")

    return sorted(matching_products)

# =====================
# Ejecución principal
# =====================


# Cargar las reglas de FP-Growth
rules = load_model('parcial/fp_growth/rules_fp_growth.pkl')

if rules is not None:
    # Mostrar estadísticas generales
    show_general_stats(rules)

    # Ejemplo de uso con un producto específico
    print("\n" + "="*60)
    producto = 'WOODEN TREE CHRISTMAS SCANDINAVIAN'  # Cambiar por un producto válido
    recommendations = get_product_recommendations(producto, rules)

    # Ejemplo adicional
    print("\n" + "="*60)
    producto2 = 'JUMBO BAG PINK POLKADOT'  # Otro ejemplo
    recommendations2 = get_product_recommendations(producto2, rules)

else:
    print("No se pudo cargar el modelo. Ejecuta primero fp_growth.py para generar las reglas.")
