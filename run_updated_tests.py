#!/usr/bin/env python3
"""
Script para ejecutar todos los tests actualizados y verificar que funcionan con la nueva versión de la librería.
"""

import subprocess
import sys
import os

def run_tests():
    """Ejecutar todos los tests actualizados."""
    print("🧪 EJECUTANDO TESTS ACTUALIZADOS")
    print("="*60)
    
    # Lista de archivos de test actualizados
    test_files = [
        "tests/test_fixes_verification.py",
        "tests/test_plot_types_unit.py", 
        "tests/test_matplotlib_analyzer.py",
        "tests/test_seaborn_analyzer.py",
        "tests/test_converter.py",
        "tests/test_matplotlib_formats.py",
        "tests/test_advanced_integration.py",
        "tests/test_base_analyzer.py",
        "tests/test_utils.py"
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\n📁 Ejecutando: {test_file}")
            print("-" * 50)
            
            try:
                # Ejecutar el test con pytest
                result = subprocess.run([
                    sys.executable, "-m", "pytest", test_file, 
                    "-v", "--tb=short", "--no-header"
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print("✅ PASSED")
                    # Contar tests pasados
                    output_lines = result.stdout.split('\n')
                    for line in output_lines:
                        if 'passed' in line and 'failed' in line:
                            # Extraer números de la línea de resumen
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == 'passed':
                                    if i > 0 and parts[i-1].isdigit():
                                        passed_tests += int(parts[i-1])
                                elif part == 'failed':
                                    if i > 0 and parts[i-1].isdigit():
                                        failed_tests += int(parts[i-1])
                else:
                    print("❌ FAILED")
                    print("Error output:")
                    print(result.stderr)
                    failed_tests += 1
                    
            except subprocess.TimeoutExpired:
                print("⏰ TIMEOUT - Test tardó demasiado")
                failed_tests += 1
            except Exception as e:
                print(f"💥 ERROR: {e}")
                failed_tests += 1
        else:
            print(f"⚠️  Archivo no encontrado: {test_file}")
    
    # Resumen final
    print("\n" + "="*60)
    print("📊 RESUMEN FINAL")
    print("="*60)
    print(f"Tests pasados: {passed_tests}")
    print(f"Tests fallidos: {failed_tests}")
    print(f"Total: {passed_tests + failed_tests}")
    
    if failed_tests == 0:
        print("\n🎉 ¡TODOS LOS TESTS PASARON!")
        print("✅ La librería está funcionando correctamente con la nueva estructura")
        return True
    else:
        print(f"\n⚠️  {failed_tests} tests fallaron")
        print("❌ Hay problemas que necesitan ser corregidos")
        return False

def run_specific_tests():
    """Ejecutar tests específicos para verificar funcionalidades clave."""
    print("\n🔍 EJECUTANDO TESTS ESPECÍFICOS")
    print("="*60)
    
    # Tests específicos para verificar funcionalidades clave
    specific_tests = [
        "tests/test_fixes_verification.py::TestFixesVerification::test_normal_distribution_detection_fix",
        "tests/test_fixes_verification.py::TestFixesVerification::test_multimodal_distribution_detection_fix",
        "tests/test_fixes_verification.py::TestFixesVerification::test_trimodal_distribution_detection_fix",
        "tests/test_plot_types_unit.py::TestPlotTypesUnit::test_histogram_distribution_detection",
        "tests/test_plot_types_unit.py::TestPlotTypesUnit::test_multimodal_distribution_detection",
        "tests/test_plot_types_unit.py::TestPlotTypesUnit::test_trimodal_distribution_detection"
    ]
    
    for test in specific_tests:
        print(f"\n🧪 Ejecutando: {test}")
        print("-" * 40)
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test, 
                "-v", "--tb=short", "--no-header"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ PASSED")
            else:
                print("❌ FAILED")
                print("Error output:")
                print(result.stderr)
                
        except Exception as e:
            print(f"💥 ERROR: {e}")

if __name__ == "__main__":
    print("🚀 INICIANDO VERIFICACIÓN DE TESTS ACTUALIZADOS")
    print("="*60)
    
    # Ejecutar todos los tests
    success = run_tests()
    
    # Ejecutar tests específicos
    run_specific_tests()
    
    print("\n" + "="*60)
    if success:
        print("🎉 VERIFICACIÓN COMPLETADA EXITOSAMENTE")
        print("✅ Todos los tests están actualizados y funcionando")
    else:
        print("⚠️  VERIFICACIÓN COMPLETADA CON PROBLEMAS")
        print("❌ Algunos tests necesitan más ajustes")
    
    print("="*60) 