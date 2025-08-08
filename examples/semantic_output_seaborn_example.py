#!/usr/bin/env python3
"""
Ejemplo completo de output semántico para seaborn.

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
import seaborn as sns

from plot2llm import FigureConverter


def create_seaborn_examples():
    """Crear ejemplos de seaborn y mostrar output semántico completo."""

    print("=" * 80)
    print("EJEMPLO DE OUTPUT SEMÁNTICO - SEABORN")
    print("=" * 80)

    # Configurar el converter
    converter = FigureConverter()

    # Configurar estilo de seaborn
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # ============================================================================
    # 1. LINE PLOT
    # ============================================================================
    print("\n" + "=" * 60)
    print("1. LINE PLOT")
    print("=" * 60)

    # Crear datos
    x = np.linspace(0, 10, 50)
    y = np.sin(x) + np.random.normal(0, 0.1, 50)

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=x, y=y, ax=ax, color="blue", linewidth=2, marker="o", markersize=6)
    ax.set_title("Sinusoidal Pattern Analysis", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)

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
    y_scatter = 0.3 * x_scatter + np.random.normal(0, 0.2, 100)

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=x_scatter, y=y_scatter, ax=ax, color="red", alpha=0.6, s=50)
    ax.set_title("Correlation Analysis with Seaborn", fontsize=14, fontweight="bold")
    ax.set_xlabel("Variable X", fontsize=12)
    ax.set_ylabel("Variable Y", fontsize=12)

    # Agregar línea de regresión
    sns.regplot(
        x=x_scatter,
        y=y_scatter,
        ax=ax,
        scatter=False,
        color="blue",
        line_kws={"linewidth": 2},
    )

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
    values = [15, 32, 18, 45, 22, 38]

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=categories, y=values, ax=ax, color="green", alpha=0.7)
    ax.set_title("Categorical Analysis with Seaborn", fontsize=14, fontweight="bold")
    ax.set_xlabel("Categories", fontsize=12)
    ax.set_ylabel("Values", fontsize=12)

    # Agregar valores en las barras
    for i, v in enumerate(values):
        ax.text(i, v + 1, str(v), ha="center", va="bottom", fontweight="bold")

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
    sns.histplot(data, bins=30, ax=ax, color="purple", alpha=0.7, edgecolor="black")
    ax.set_title("Distribution Analysis with Seaborn", fontsize=14, fontweight="bold")
    ax.set_xlabel("Values", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)

    # Agregar línea de densidad
    sns.kdeplot(data, ax=ax, color="red", linewidth=2)

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
    create_seaborn_examples()
