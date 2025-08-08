#!/usr/bin/env python3
"""
Ejemplo completo de output semántico para matplotlib.

Este script demuestra el output semántico completo para 4 tipos de gráficos:
- Line plot
- Scatter plot
- Bar plot
- Histogram

El output incluye todos los campos del formato semántico.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from plot2llm import FigureConverter


def create_matplotlib_examples():
    """Crear ejemplos de matplotlib y mostrar output semántico completo."""

    print("=" * 80)
    print("EJEMPLO DE OUTPUT SEMÁNTICO - MATPLOTLIB")
    print("=" * 80)

    # Configurar el converter
    converter = FigureConverter()

    # ============================================================================
    # 1. LINE PLOT
    # ============================================================================
    print("\n" + "=" * 60)
    print("1. LINE PLOT")
    print("=" * 60)

    # Crear datos
    x = np.linspace(0, 10, 50)
    y = 2 * x + 1 + np.random.normal(0, 0.5, 50)

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, "bo-", linewidth=2, markersize=6, label="Linear Trend")
    ax.set_title("Linear Relationship Example", fontsize=14, fontweight="bold")
    ax.set_xlabel("X Values", fontsize=12)
    ax.set_ylabel("Y Values", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Analizar
    result = converter.convert(fig, "semantic")

    # Mostrar output completo
    print("\n--- OUTPUT SEMÁNTICO COMPLETO ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    plt.close(fig)

    # ============================================================================
    # 2. SCATTER PLOT
    # ============================================================================
    print("\n" + "=" * 60)
    print("2. SCATTER PLOT")
    print("=" * 60)

    # Crear datos
    x_scatter = np.random.normal(0, 1, 100)
    y_scatter = 0.5 * x_scatter + np.random.normal(0, 0.3, 100)

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    _ = ax.scatter(x_scatter, y_scatter, alpha=0.6, color="red", s=50)
    ax.set_title("Correlation Analysis", fontsize=14, fontweight="bold")
    ax.set_xlabel("Variable X", fontsize=12)
    ax.set_ylabel("Variable Y", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Analizar
    result = converter.convert(fig, "semantic")

    # Mostrar output completo
    print("\n--- OUTPUT SEMÁNTICO COMPLETO ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    plt.close(fig)

    # ============================================================================
    # 3. BAR PLOT
    # ============================================================================
    print("\n" + "=" * 60)
    print("3. BAR PLOT")
    print("=" * 60)

    # Crear datos
    categories = ["A", "B", "C", "D", "E", "F"]
    values = [23, 45, 12, 36, 28, 19]

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, values, color="green", alpha=0.7, edgecolor="black")
    ax.set_title("Categorical Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Categories", fontsize=12)
    ax.set_ylabel("Values", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    # Agregar valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    # Analizar
    result = converter.convert(fig, "semantic")

    # Mostrar output completo
    print("\n--- OUTPUT SEMÁNTICO COMPLETO ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    plt.close(fig)

    # ============================================================================
    # 4. HISTOGRAM
    # ============================================================================
    print("\n" + "=" * 60)
    print("4. HISTOGRAM")
    print("=" * 60)

    # Crear datos
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(
        data, bins=30, alpha=0.7, color="purple", edgecolor="black"
    )
    ax.set_title("Normal Distribution Analysis", fontsize=14, fontweight="bold")
    ax.set_xlabel("Values", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Agregar línea de densidad
    from scipy.stats import norm

    x = np.linspace(data.min(), data.max(), 100)
    ax.plot(
        x,
        norm.pdf(x, data.mean(), data.std()) * len(data) * (bins[1] - bins[0]),
        "r-",
        linewidth=2,
        label="Normal Distribution",
    )
    ax.legend()

    # Analizar
    result = converter.convert(fig, "semantic")

    # Mostrar output completo
    print("\n--- OUTPUT SEMÁNTICO COMPLETO ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    plt.close(fig)

    # ============================================================================
    # OPCIONES ALTERNATIVAS (COMENTADAS)
    # ============================================================================
    print("\n" + "=" * 60)
    print("OPCIONES ALTERNATIVAS DE OUTPUT")
    print("=" * 60)

    # Para generar output en formato TEXT:
    # result_text = converter.convert(fig, 'text')
    # print("--- OUTPUT TEXT ---")
    # print(result_text)

    # Para generar output en formato JSON:
    # result_json = converter.convert(fig, 'json')
    # print("--- OUTPUT JSON ---")
    # print(json.dumps(result_json, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("EJEMPLO COMPLETADO")
    print("=" * 80)


if __name__ == "__main__":
    create_matplotlib_examples()
