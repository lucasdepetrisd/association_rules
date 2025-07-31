"""
Módulo para capturar y gestionar la salida de los scripts de minería de datos.
Permite capturar todos los prints y guardarlos en archivos para logging.
"""

import sys
import io
from typing import Optional


class TeeOutput:
    """
    Clase para capturar output de print y mantener salida en consola.
    
    Esta clase permite:
    - Capturar toda la salida de print() en un buffer
    - Mantener la salida normal en consola
    - Recuperar todo el texto capturado
    """
    
    def __init__(self):
        self.terminal = sys.stdout
        self.log = io.StringIO()
        self._original_stdout = None
    
    def write(self, message):
        """Escribe el mensaje tanto en terminal como en buffer"""
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        """Flush de ambos outputs"""
        self.terminal.flush()
        self.log.flush()
    
    def get_output(self):
        """Obtiene toda la salida capturada como string"""
        return self.log.getvalue()
    
    def start_capture(self):
        """Inicia la captura de stdout"""
        self._original_stdout = sys.stdout
        sys.stdout = self
        return self
    
    def stop_capture(self):
        """Detiene la captura y restaura stdout original"""
        if self._original_stdout:
            sys.stdout = self._original_stdout
            self._original_stdout = None


def setup_output_capture():
    """
    Configura y activa la captura de salida.
    
    Returns:
        TeeOutput: Instancia configurada para capturar salida
        
    Usage:
        output_capture = setup_output_capture()
        # ... tu código aquí ...
        captured_text = finalize_output_capture(output_capture, "mi_archivo.txt")
    """
    capture = TeeOutput()
    capture.start_capture()
    return capture


def finalize_output_capture(capture: TeeOutput, output_filename: str) -> str:
    """
    Finaliza la captura, guarda en archivo y retorna el texto capturado.
    
    Args:
        capture: Instancia de TeeOutput activa
        output_filename: Ruta donde guardar el log
        
    Returns:
        str: Todo el texto capturado durante la ejecución
    """
    # Detener captura y restaurar stdout
    capture.stop_capture()
    
    # Obtener texto capturado
    captured_output = capture.get_output()
    
    # Guardar en archivo
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(captured_output)
    
    print(f"Salida de ejecución guardada en: {output_filename}")
    
    return captured_output


def print_script_header(script_name: str, algorithm_name: str):
    """
    Imprime un header estándar para los scripts.
    
    Args:
        script_name: Nombre del script (ej: "APRIORI", "FP-GROWTH")
        algorithm_name: Nombre del algoritmo para mostrar
    """
    import pandas as pd
    
    print(f"=== INICIO DE EJECUCIÓN {script_name.upper()} ===")
    print(f"Algoritmo: {algorithm_name}")
    print(f"Timestamp: {pd.Timestamp.now()}")
    print("="*50)


def print_script_footer(script_name: str):
    """
    Imprime un footer estándar para los scripts.
    
    Args:
        script_name: Nombre del script (ej: "APRIORI", "FP-GROWTH")
    """
    import pandas as pd
    
    print(f"\n=== FIN DE EJECUCIÓN {script_name.upper()} ===")
    print(f"Timestamp: {pd.Timestamp.now()}")
    print("="*50)


# Context manager para uso más elegante
class OutputCapture:
    """
    Context manager para captura de salida.
    
    Usage:
        with OutputCapture("mi_log.txt") as capture:
            print("Este texto será capturado")
            # ... más código ...
        # Al salir del context, se guarda automáticamente
    """
    
    def __init__(self, output_filename: str):
        self.output_filename = output_filename
        self.capture = None
        
    def __enter__(self):
        self.capture = setup_output_capture()
        return self.capture
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.capture:
            finalize_output_capture(self.capture, self.output_filename)
