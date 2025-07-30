"""
Ejemplo de uso del ResourceMonitor en cualquier script
"""

from resource_monitor import ResourceMonitor, get_dataset_memory_usage, print_initial_system_info
import pandas as pd
import numpy as np
import time


def ejemplo_uso_basico():
    """Ejemplo básico de uso del ResourceMonitor"""

    # Crear instancia del monitor
    monitor = ResourceMonitor(monitoring_interval=0.5)

    # Iniciar monitoreo
    initial_info = monitor.start_monitoring()
    print_initial_system_info(initial_info)

    print("\n=== SIMULANDO PROCESO INTENSIVO ===")

    # Simular proceso que consume memoria y CPU
    data = []
    for i in range(1000):
        # Operación que consume CPU y memoria
        arr = np.random.random((1000, 100))
        result = np.dot(arr, arr.T)
        data.append(result)

        if i % 100 == 0:
            print(f"Iteración {i}/1000")

        time.sleep(0.01)  # Pequeña pausa

    print("Proceso completado!")

    # Detener monitoreo y obtener métricas
    metrics = monitor.stop_monitoring()

    # Mostrar resumen
    monitor.print_summary(metrics)
    monitor.print_progression_sample(sample_every=20)

    return metrics


def ejemplo_con_dataframes():
    """Ejemplo usando DataFrames de pandas"""

    monitor = ResourceMonitor()
    initial_info = monitor.start_monitoring()

    print_initial_system_info(initial_info)

    # Crear DataFrame grande
    print("\n=== CREANDO DATAFRAME GRANDE ===")
    df = pd.DataFrame(np.random.random((100000, 50)))
    get_dataset_memory_usage(df, "DataFrame original")

    # Operaciones con el DataFrame
    print("Realizando operaciones...")
    df_processed = df.groupby(df.iloc[:, 0] > 0.5).agg(
        ['mean', 'std', 'min', 'max'])
    get_dataset_memory_usage(df_processed, "DataFrame procesado")

    # Más operaciones
    correlation_matrix = df.corr()
    get_dataset_memory_usage(correlation_matrix, "Matriz de correlación")

    # Detener y mostrar métricas
    metrics = monitor.stop_monitoring()
    monitor.print_summary(metrics)

    return metrics


def ejemplo_comparacion_algoritmos():
    """Ejemplo comparando dos algoritmos diferentes"""

    def algoritmo_ineficiente(data):
        result = []
        for i in range(len(data)):
            for j in range(len(data)):
                result.append(data[i] * data[j])
        return result

    def algoritmo_eficiente(data):
        data_array = np.array(data)
        return np.outer(data_array, data_array).flatten()

    data = list(range(1000))

    # Probar algoritmo ineficiente
    print("=== ALGORITMO INEFICIENTE ===")
    monitor1 = ResourceMonitor()
    monitor1.start_monitoring()

    result1 = algoritmo_ineficiente(data)
    metrics1 = monitor1.stop_monitoring()

    print("Algoritmo ineficiente completado")
    monitor1.print_summary(metrics1)

    # Probar algoritmo eficiente
    print("\n=== ALGORITMO EFICIENTE ===")
    monitor2 = ResourceMonitor()
    monitor2.start_monitoring()

    result2 = algoritmo_eficiente(data)
    metrics2 = monitor2.stop_monitoring()

    print("Algoritmo eficiente completado")
    monitor2.print_summary(metrics2)

    # Comparar resultados
    print(f"\n=== COMPARACIÓN ===")
    print(f"Tiempo ineficiente: {metrics1['execution_time_seconds']:.2f}s")
    print(f"Tiempo eficiente: {metrics2['execution_time_seconds']:.2f}s")
    print(
        f"Mejora de velocidad: {metrics1['execution_time_seconds'] / metrics2['execution_time_seconds']:.1f}x")

    print(
        f"Memoria pico ineficiente: {metrics1['memory_peak_process_MB']:.2f} MB")
    print(
        f"Memoria pico eficiente: {metrics2['memory_peak_process_MB']:.2f} MB")


if __name__ == "__main__":
    print("=== EJEMPLO 1: USO BÁSICO ===")
    ejemplo_uso_basico()

    print("\n" + "="*60)
    print("=== EJEMPLO 2: CON DATAFRAMES ===")
    ejemplo_con_dataframes()

    print("\n" + "="*60)
    print("=== EJEMPLO 3: COMPARACIÓN DE ALGORITMOS ===")
    ejemplo_comparacion_algoritmos()
