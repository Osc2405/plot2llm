import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plot2llm import FigureAnalyzer
from plot2llm.formatters import SemanticFormatter
import json

def print_section(title, section):
    """Utility function to print sections clearly"""
    print(f"\n=== {title} ===")
    print(json.dumps(section, indent=2))

def test_semantic_output():
    analyzer = FigureAnalyzer()
    formatter = SemanticFormatter()
    
    # Test 1: Matplotlib - Linear Business Data
    print("\n" + "="*50)
    print("MATPLOTLIB TEST - Business Revenue Growth")
    print("="*50)
    
    # Crear datos de ejemplo (lineal con un poco de ruido)
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y = 2 * x + 1 + np.random.normal(0, 0.5, 20)  # y = 2x + 1 con ruido
    
    # Crear gráfico matplotlib
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x, y, 'bo-', label='Revenue Growth')
    ax1.set_title('Business Revenue Growth Over Time')
    ax1.set_xlabel('Time (months)')
    ax1.set_ylabel('Revenue (millions)')
    ax1.grid(True)
    ax1.legend()
    
    # Analizar y formatear
    analysis_mpl = analyzer.analyze(fig1, figure_type="matplotlib")
    semantic_output_mpl = formatter.format(analysis_mpl)
    
    # Mostrar todas las secciones
    sections = [
        "metadata",
        "axes",
        "layout",
        "data_summary",
        "statistical_insights",
        "pattern_analysis",
        "visual_elements",
        "domain_context",
        "llm_description",
        "llm_context"
    ]
    
    for section in sections:
        if section in semantic_output_mpl:
            print_section(section.upper(), semantic_output_mpl[section])
    
    plt.close(fig1)
    
    # Test 2: Seaborn - Scatter Plot
    print("\n" + "="*50)
    print("SEABORN TEST - Customer Satisfaction vs Experience")
    print("="*50)
    
    # Crear datos para seaborn (relación no lineal)
    np.random.seed(42)
    n_points = 30
    x_sns = np.linspace(0, 10, n_points)
    y_sns = 0.5 * x_sns**2 - 2 * x_sns + np.random.normal(0, 3, n_points)
    
    # Crear gráfico seaborn
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=x_sns, y=y_sns, ax=ax2)
    ax2.set_title('Customer Satisfaction vs Experience Level')
    ax2.set_xlabel('Years of Experience')
    ax2.set_ylabel('Satisfaction Score')
    
    # Analizar y formatear
    analysis_sns = analyzer.analyze(fig2, figure_type="seaborn")
    semantic_output_sns = formatter.format(analysis_sns)
    
    # Mostrar todas las secciones
    for section in sections:
        if section in semantic_output_sns:
            print_section(section.upper(), semantic_output_sns[section])
    
    plt.close(fig2)

if __name__ == "__main__":
    test_semantic_output() 