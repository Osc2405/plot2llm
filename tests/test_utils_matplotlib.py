import matplotlib.pyplot as plt
import numpy as np
from plot2llm.utils_matplotlib import identify_plot_type, extract_axes_section
from plot2llm.analyzers.matplotlib_analyzer import MatplotlibAnalyzer

def test_identify_plot_type_line():
    fig, ax = plt.subplots()
    x = np.arange(5)
    y = x ** 2
    ax.plot(x, y)
    assert identify_plot_type(ax) == "line"
    plt.close(fig)

def test_identify_plot_type_bar():
    fig, ax = plt.subplots()
    categories = ['A', 'B', 'C']
    values = [10, 20, 15]
    ax.bar(categories, values)
    assert identify_plot_type(ax) == "bar"
    plt.close(fig)

def test_extract_axes_section_bar():
    fig, ax = plt.subplots()
    categories = ['A', 'B', 'C']
    values = [10, 20, 15]
    ax.bar(categories, values)
    section = extract_axes_section(ax)
    assert section["plot_type"] == "bar"
    assert section["x_label"] == ""
    # Puede incluir un "" extra por matplotlib, así que solo verifica que estén las categorías
    for cat in categories:
        assert cat in section["categories"]
    plt.close(fig)

def test_extract_axes_section_line():
    fig, ax = plt.subplots()
    x = np.arange(5)
    y = x ** 2
    ax.plot(x, y, label="Cuadrática")
    section = extract_axes_section(ax)
    assert section["plot_type"] == "line"
    assert section["lines"][0]["label"] == "Cuadrática"
    assert section["lines"][0]["xdata"] == list(x)
    assert section["lines"][0]["ydata"] == list(y)
    plt.close(fig)

def test_categorical_bar_statistics():
    """Prueba las estadísticas para gráficos de barras categóricos."""
    # Crear gráfico de barras
    fig, ax = plt.subplots()
    categories = ['A', 'B', 'C']
    values = [10, 20, 15]
    ax.bar(categories, values)
    
    # Analizar con MatplotlibAnalyzer
    analyzer = MatplotlibAnalyzer()
    result = analyzer.analyze(fig)
    
    # Verificar tipo de datos
    assert "bar_plot" in result["statistics"]["per_axis"][0]["data_types"]
    
    # Verificar información por categoría
    per_curve = result["statistics"]["per_curve"]
    assert len(per_curve) == len(categories)
    
    # Verificar que los porcentajes suman 100%
    total_percentage = sum(curve["percentage"] for curve in per_curve)
    assert abs(total_percentage - 100.0) < 0.01
    
    # Verificar valores específicos
    values_dict = {curve["category"]: curve["value"] for curve in per_curve}
    assert values_dict["A"] == 10
    assert values_dict["B"] == 20
    assert values_dict["C"] == 15
    
    plt.close(fig) 