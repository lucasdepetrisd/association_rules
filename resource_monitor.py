"""
Módulo para monitoreo de recursos (memoria y CPU) en tiempo real
Útil para análisis de rendimiento de algoritmos de machine learning
"""

import time
import psutil
import threading
import gc
from typing import Dict, List, Optional


class ResourceMonitor:
    """
    Monitor de recursos que rastrea memoria y CPU en tiempo real
    """
    
    def __init__(self, monitoring_interval: float = 0.5):
        """
        Inicializar el monitor de recursos
        
        Args:
            monitoring_interval: Intervalo de monitoreo en segundos (default: 0.5)
        """
        self.monitoring_interval = monitoring_interval
        self.process = psutil.Process()
        self.start_time = None
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Variables para almacenar métricas
        self.memory_logs = []
        self.peak_process_memory = 0
        self.peak_system_memory = 0
        self.peak_cpu_percent = 0
        self.total_cpu_time = 0
        
        # Memoria inicial
        self.memory_before = 0
        self.system_memory_before = 0
        
    def _monitor_resources(self):
        """Función interna que monitorea recursos en tiempo real"""
        while self.monitoring_active:
            try:
                # Memoria del proceso (RSS + VMS)
                mem_info = self.process.memory_info()
                process_rss = mem_info.rss / (1024 * 1024)  # Memoria residente
                process_vms = mem_info.vms / (1024 * 1024)  # Memoria virtual
                
                # Memoria del sistema
                system_memory = psutil.virtual_memory()
                system_used = system_memory.used / (1024 * 1024)
                system_percent = system_memory.percent
                
                # CPU del proceso y del sistema
                process_cpu_percent = self.process.cpu_percent()  # % de CPU del proceso
                system_cpu_percent = psutil.cpu_percent()   # % de CPU del sistema
                cpu_times = self.process.cpu_times()
                current_cpu_time = cpu_times.user + cpu_times.system  # Tiempo total de CPU usado
                
                # Actualizar picos
                self.peak_process_memory = max(self.peak_process_memory, process_rss)
                self.peak_system_memory = max(self.peak_system_memory, system_used)
                self.peak_cpu_percent = max(self.peak_cpu_percent, process_cpu_percent)
                self.total_cpu_time = current_cpu_time
                
                # Log detallado
                self.memory_logs.append({
                    'timestamp': time.time() - self.start_time,
                    'process_rss_MB': process_rss,
                    'process_vms_MB': process_vms,
                    'system_used_MB': system_used,
                    'system_percent': system_percent,
                    'process_cpu_percent': process_cpu_percent,
                    'system_cpu_percent': system_cpu_percent,
                    'cpu_time_seconds': current_cpu_time
                })
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Error en monitoreo: {e}")
                break
    
    def start_monitoring(self, clean_memory: bool = True) -> Dict:
        """
        Iniciar el monitoreo de recursos
        
        Args:
            clean_memory: Si hacer garbage collection antes de empezar
            
        Returns:
            Dict con información inicial del sistema
        """
        if clean_memory:
            gc.collect()
        
        self.start_time = time.time()
        self.monitoring_active = True
        
        # Memoria inicial del proceso y sistema
        self.memory_before = self.process.memory_info().rss / (1024 * 1024)
        self.system_memory_before = psutil.virtual_memory().used / (1024 * 1024)
        self.peak_process_memory = self.memory_before
        self.peak_system_memory = self.system_memory_before
        
        # Información inicial del sistema
        initial_info = {
            'memory_process_rss_MB': self.memory_before,
            'memory_process_vms_MB': self.process.memory_info().vms / (1024 * 1024),
            'memory_system_used_MB': self.system_memory_before,
            'memory_system_percent': psutil.virtual_memory().percent,
            'cpu_system_percent': psutil.cpu_percent(),
            'cpu_cores_physical': psutil.cpu_count(),
            'cpu_cores_logical': psutil.cpu_count(logical=True)
        }
        
        # Iniciar monitoreo en hilo separado
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
        
        return initial_info
    
    def stop_monitoring(self, clean_memory: bool = True) -> Dict:
        """
        Detener el monitoreo y retornar métricas finales
        
        Args:
            clean_memory: Si hacer garbage collection al final
            
        Returns:
            Dict con todas las métricas calculadas
        """
        # Detener monitoreo
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
        
        if clean_memory:
            gc.collect()
        
        # Calcular tiempo total de ejecución
        execution_time = time.time() - self.start_time
        
        # Memoria final
        memory_final = self.process.memory_info().rss / (1024 * 1024)
        system_memory_final = psutil.virtual_memory().used / (1024 * 1024)
        
        # Calcular métricas basadas en la progresión
        metrics = self._calculate_metrics(execution_time, memory_final, system_memory_final)
        
        return metrics
    
    def _calculate_metrics(self, execution_time: float, memory_final: float, system_memory_final: float) -> Dict:
        """Calcular todas las métricas basadas en los logs"""
        
        if self.memory_logs:
            # Obtener valores de todos los logs
            process_memory_values = [log['process_rss_MB'] for log in self.memory_logs]
            system_memory_values = [log['system_used_MB'] for log in self.memory_logs]
            system_percent_values = [log['system_percent'] for log in self.memory_logs]
            process_cpu_values = [log['process_cpu_percent'] for log in self.memory_logs]
            system_cpu_values = [log['system_cpu_percent'] for log in self.memory_logs]
            cpu_time_values = [log['cpu_time_seconds'] for log in self.memory_logs]
            
            # Métricas de memoria
            peak_process_from_logs = max(process_memory_values)
            peak_system_from_logs = max(system_memory_values)
            max_system_percent = max(system_percent_values)
            initial_process = process_memory_values[0]
            final_process = process_memory_values[-1]
            memory_freed = peak_process_from_logs - final_process
            
            # Métricas de CPU
            peak_process_cpu = max(process_cpu_values)
            peak_system_cpu = max(system_cpu_values)
            avg_process_cpu = sum(process_cpu_values) / len(process_cpu_values)
            avg_system_cpu = sum(system_cpu_values) / len(system_cpu_values)
            total_cpu_time_used = max(cpu_time_values) - min(cpu_time_values)
            
        else:
            # Fallback si no hay logs
            peak_process_from_logs = max(self.memory_before, memory_final)
            peak_system_from_logs = self.peak_system_memory
            max_system_percent = 0
            initial_process = self.memory_before
            final_process = memory_final
            memory_freed = peak_process_from_logs - final_process
            peak_process_cpu = 0
            peak_system_cpu = 0
            avg_process_cpu = 0
            avg_system_cpu = 0
            total_cpu_time_used = 0
        
        return {
            'execution_time_seconds': execution_time,
            'total_samples': len(self.memory_logs),
            
            # Métricas de memoria
            'memory_initial_MB': initial_process,
            'memory_final_MB': final_process,
            'memory_peak_process_MB': peak_process_from_logs,
            'memory_peak_system_MB': peak_system_from_logs,
            'memory_max_additional_MB': peak_process_from_logs - initial_process,
            'memory_freed_after_peak_MB': memory_freed,
            'max_system_percent': max_system_percent,
            
            # Métricas de CPU
            'cpu_peak_process_percent': peak_process_cpu,
            'cpu_peak_system_percent': peak_system_cpu,
            'cpu_avg_process_percent': avg_process_cpu,
            'cpu_avg_system_percent': avg_system_cpu,
            'cpu_time_used_seconds': total_cpu_time_used,
            'cpu_efficiency_percent': (total_cpu_time_used / execution_time) * 100 if execution_time > 0 else 0,
            
            # Logs completos para análisis detallado
            'logs': self.memory_logs
        }
    
    def print_summary(self, metrics: Optional[Dict] = None):
        """
        Imprimir un resumen de las métricas
        
        Args:
            metrics: Dict de métricas (si no se proporciona, se calculan automáticamente)
        """
        if metrics is None:
            metrics = self.stop_monitoring()
        
        print(f"\n=== RESUMEN COMPLETO DE RECURSOS ===")
        print(f"Tiempo de ejecución: {metrics['execution_time_seconds']:.2f} segundos")
        print(f"Total de muestras capturadas: {metrics['total_samples']}")
        
        print(f"\n--- MÉTRICAS DE MEMORIA ---")
        print(f"🔥 Pico de memoria del proceso: {metrics['memory_peak_process_MB']:.2f} MB")
        print(f"🔥 Pico de memoria del sistema: {metrics['memory_peak_system_MB']:.2f} MB")
        print(f"🔥 Porcentaje máximo de RAM usado: {metrics['max_system_percent']:.1f}%")
        print(f"📊 Memoria inicial del proceso: {metrics['memory_initial_MB']:.2f} MB")
        print(f"📊 Memoria final del proceso: {metrics['memory_final_MB']:.2f} MB")
        print(f"📈 Memoria máxima adicional usada: {metrics['memory_max_additional_MB']:.2f} MB")
        print(f"🗑️ Memoria liberada después del pico: {metrics['memory_freed_after_peak_MB']:.2f} MB")
        
        print(f"\n--- MÉTRICAS DE CPU ---")
        print(f"🚀 Pico de CPU del proceso: {metrics['cpu_peak_process_percent']:.1f}%")
        print(f"🚀 Pico de CPU del sistema: {metrics['cpu_peak_system_percent']:.1f}%")
        print(f"📊 Promedio de CPU del proceso: {metrics['cpu_avg_process_percent']:.1f}%")
        print(f"📊 Promedio de CPU del sistema: {metrics['cpu_avg_system_percent']:.1f}%")
        print(f"⏱️ Tiempo total de CPU usado: {metrics['cpu_time_used_seconds']:.2f} segundos")
        print(f"⚡ Eficiencia CPU: {metrics['cpu_efficiency_percent']:.1f}% (CPU time / Wall time)")
    
    def print_progression_sample(self, sample_every: int = 10):
        """
        Imprimir una muestra de la progresión de recursos
        
        Args:
            sample_every: Mostrar cada N registros
        """
        print(f"\n--- PROGRESIÓN DE MEMORIA Y CPU (muestras cada {sample_every} registros) ---")
        for i, log in enumerate(self.memory_logs[::sample_every]):
            print(f"T+{log['timestamp']:.1f}s: Proceso={log['process_rss_MB']:.1f}MB, "
                  f"Sistema={log['system_used_MB']:.0f}MB ({log['system_percent']:.1f}%), "
                  f"CPU_proc={log['process_cpu_percent']:.1f}%, CPU_sys={log['system_cpu_percent']:.1f}%")


def get_dataset_memory_usage(df, name: str = "DataFrame") -> float:
    """
    Obtener el uso de memoria de un DataFrame
    
    Args:
        df: DataFrame de pandas
        name: Nombre descriptivo del DataFrame
        
    Returns:
        Uso de memoria en MB
    """
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"Tamaño de {name}: {memory_mb:.2f} MB")
    return memory_mb


def print_initial_system_info(initial_info: Dict):
    """
    Imprimir información inicial del sistema
    
    Args:
        initial_info: Dict con información inicial del sistema
    """
    print(f"=== ESTADO INICIAL ===")
    print(f"Memoria proceso (RSS): {initial_info['memory_process_rss_MB']:.2f} MB")
    print(f"Memoria proceso (VMS): {initial_info['memory_process_vms_MB']:.2f} MB")
    print(f"Memoria sistema total: {initial_info['memory_system_used_MB']:.2f} MB ({initial_info['memory_system_percent']:.1f}%)")
    print(f"CPU actual del sistema: {initial_info['cpu_system_percent']:.1f}%")
    print(f"Número de núcleos CPU: {initial_info['cpu_cores_physical']} físicos, {initial_info['cpu_cores_logical']} lógicos")
