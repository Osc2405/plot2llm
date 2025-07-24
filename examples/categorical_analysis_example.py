import matplotlib.pyplot as plt
import numpy as np
import json
from plot2llm.analyzers.matplotlib_analyzer import MatplotlibAnalyzer

def create_sales_chart():
    """
    Crea un gráfico de barras realista con datos de ventas por categoría.
    """
    # Datos de ejemplo: Ventas por categoría de producto
    categories = ['Electrónicos', 'Ropa', 'Hogar', 'Deportes', 'Libros']
    sales = [120000, 85000, 67000, 45000, 32000]

    # Crear figura con estilo moderno
    plt.style.use('bmh')  # Usando un estilo incorporado de matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))

    # Crear gráfico de barras con colores personalizados
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f', '#9b59b6']
    bars = ax.bar(categories, sales, color=colors)

    # Personalizar el gráfico
    ax.set_title('Ventas por Categoría - 2023', fontsize=14, pad=20)
    ax.set_xlabel('Categoría de Producto', fontsize=12)
    ax.set_ylabel('Ventas (USD)', fontsize=12)
    
    # Agregar valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom')

    # Personalizar grid y estilo
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Rotar etiquetas del eje x para mejor legibilidad
    plt.xticks(rotation=15)

    return fig

def print_section(title, content):
    """Imprime una sección del output de forma legible."""
    print(f"\n{'='*20} {title} {'='*20}")
    print(json.dumps(content, indent=2, ensure_ascii=False))

def main():
    try:
        # Crear el gráfico
        fig = create_sales_chart()
        
        # Analizar el gráfico
        analyzer = MatplotlibAnalyzer()
        result = analyzer.analyze(fig)
        
        # Imprimir resultados de forma organizada
        print("\nANÁLISIS SEMÁNTICO DEL GRÁFICO DE VENTAS")
        print("="*50)
        
        # Información de la figura
        print_section("INFORMACIÓN DE LA FIGURA", result["figure"])
        
        # Información de los ejes
        print_section("INFORMACIÓN DE LOS EJES", result["axes"])
        
        # Información de colores
        print_section("INFORMACIÓN DE COLORES", result["colors"])
        
        # Información estadística
        print_section("INFORMACIÓN ESTADÍSTICA", result["statistics"])
        
        plt.close(fig)
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        raise

if __name__ == "__main__":
    main() 