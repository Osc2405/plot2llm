"""
Demo de integración con LLMs usando plot2llm
Muestra diferentes formatos de salida y casos de uso para IA
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import plot2llm


def llm_analysis_pipeline():
    """
    Demuestra cómo integrar plot2llm con análisis de LLMs
    """
    print("=== DEMO DE INTEGRACIÓN CON LLMs ===")

    # Crear datos de ejemplo
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "edad": np.random.normal(35, 10, 1000),
            "ingresos": np.random.lognormal(10, 0.5, 1000),
            "satisfaccion": np.random.uniform(1, 10, 1000),
            "categoria": np.random.choice(["A", "B", "C"], 1000),
        }
    )

    # Crear visualización compleja
    fig = plt.figure(figsize=(16, 10))

    # Subplot 1: Distribución de edad
    plt.subplot(2, 3, 1)
    plt.hist(data["edad"], bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    plt.title("Distribución de Edad", fontweight="bold")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")

    # Subplot 2: Ingresos vs Satisfacción
    plt.subplot(2, 3, 2)
    scatter = plt.scatter(
        data["ingresos"],
        data["satisfaccion"],
        alpha=0.6,
        c=data["edad"],
        cmap="viridis",
        s=50,
    )
    plt.colorbar(scatter, label="Edad")
    plt.title("Ingresos vs Satisfacción", fontweight="bold")
    plt.xlabel("Ingresos")
    plt.ylabel("Satisfacción")

    # Subplot 3: Boxplot por categoría
    plt.subplot(2, 3, 3)
    data.boxplot(column="ingresos", by="categoria", ax=plt.gca())
    plt.title("Ingresos por Categoría", fontweight="bold")
    plt.suptitle("")  # Eliminar título automático

    # Subplot 4: Heatmap de correlaciones
    plt.subplot(2, 3, 4)
    corr_matrix = data[["edad", "ingresos", "satisfaccion"]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Matriz de Correlaciones", fontweight="bold")

    # Subplot 5: Violin plot
    plt.subplot(2, 3, 5)
    sns.violinplot(data=data, x="categoria", y="satisfaccion")
    plt.title("Satisfacción por Categoría", fontweight="bold")

    # Subplot 6: Histograma de ingresos
    plt.subplot(2, 3, 6)
    plt.hist(
        data["ingresos"], bins=50, alpha=0.7, color="lightgreen", edgecolor="black"
    )
    plt.title("Distribución de Ingresos", fontweight="bold")
    plt.xlabel("Ingresos")
    plt.ylabel("Frecuencia")

    plt.tight_layout()

    # Convertir con diferentes formatos
    print("=== ANÁLISIS COMPLETO CON PLOT2LLM ===")

    # Formato de texto
    text_result = plot2llm.convert(fig, format="text", detail_level="high")
    print("\n--- FORMATO TEXTO ---")
    print(text_result[:500] + "...")

    # Formato JSON
    json_result = plot2llm.convert(fig, format="json", detail_level="high")
    print("\n--- FORMATO JSON ---")
    print(json.dumps(json_result, indent=2)[:500] + "...")

    # Formato semántico
    semantic_result = plot2llm.convert(fig, format="semantic", detail_level="high")
    print("\n--- FORMATO SEMÁNTICO ---")
    print(json.dumps(semantic_result, indent=2)[:500] + "...")

    return fig


def llm_prompt_generation():
    """
    Genera prompts optimizados para LLMs basados en visualizaciones
    """
    print("\n=== GENERACIÓN DE PROMPTS PARA LLMs ===")

    # Crear un gráfico simple para demostración
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-x / 5)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, "b-", linewidth=2, label="Función")
    ax.set_title("Función Senoidal Amortiguada", fontweight="bold")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Amplitud")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Convertir a diferentes formatos para LLMs
    semantic_output = plot2llm.convert(fig, format="semantic", detail_level="high")

    # Generar prompts para diferentes LLMs
    prompts = {
        "GPT-4": f"""
        Analiza la siguiente visualización y proporciona insights detallados:

        {json.dumps(semantic_output, indent=2)}

        Por favor, proporciona:
        1. Descripción técnica del gráfico
        2. Patrones identificados
        3. Posibles aplicaciones
        4. Recomendaciones para análisis adicional
        """,
        "Claude": f"""
        Basándote en esta visualización:

        {json.dumps(semantic_output, indent=2)}

        Responde:
        - ¿Qué tipo de fenómeno representa este gráfico?
        - ¿Cuáles son las características más importantes?
        - ¿Qué insights se pueden extraer?
        - ¿Qué preguntas adicionales sugerirías?
        """,
        "RAG Pipeline": f"""
        Contexto de visualización para RAG:

        {json.dumps(semantic_output, indent=2)}

        Usar para:
        - Búsqueda semántica
        - Generación de respuestas
        - Análisis comparativo
        """,
    }

    print("Prompts generados para diferentes LLMs:")
    for llm, prompt in prompts.items():
        print(f"\n--- {llm} ---")
        print(prompt[:300] + "...")

    return fig


def multi_format_comparison():
    """
    Compara diferentes formatos de salida para diferentes casos de uso
    """
    print("\n=== COMPARACIÓN DE FORMATOS ===")

    # Crear un gráfico de barras simple
    categories = ["A", "B", "C", "D", "E"]
    values = [23, 45, 56, 78, 32]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(
        categories, values, color=["red", "blue", "green", "orange", "purple"]
    )
    ax.set_title("Comparación de Valores por Categoría", fontweight="bold")
    ax.set_xlabel("Categoría")
    ax.set_ylabel("Valor")

    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    # Comparar formatos
    formats = {
        "text": "Descripción narrativa para documentación",
        "json": "Estructura de datos para procesamiento programático",
        "semantic": "Análisis completo optimizado para LLMs",
    }

    print("Comparación de formatos de salida:")
    for format_type, description in formats.items():
        result = plot2llm.convert(fig, format=format_type, detail_level="medium")
        print(f"\n--- {format_type.upper()} ({description}) ---")
        if format_type in ["json", "semantic"]:
            print(json.dumps(result, indent=2)[:400] + "...")
        else:
            print(result[:400] + "...")

    return fig


def error_handling_demo():
    """
    Demuestra el manejo de errores en plot2llm
    """
    print("\n=== DEMO DE MANEJO DE ERRORES ===")

    try:
        # Crear un gráfico válido
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Gráfico de Prueba")

        # Convertir exitosamente
        result = plot2llm.convert(fig, format="text")
        print("✅ Conversión exitosa:")
        print(result[:200] + "...")

    except Exception as e:
        print(f"❌ Error en conversión: {e}")

    try:
        # Intentar convertir algo que no es una figura
        result = plot2llm.convert("no es una figura", format="text")
    except Exception as e:
        print(f"✅ Error capturado correctamente: {type(e).__name__}")

    return fig


if __name__ == "__main__":
    print("DEMO DE INTEGRACIÓN CON LLMs USANDO PLOT2LLM")
    print("=" * 60)

    # Ejecutar demos
    llm_analysis_pipeline()
    llm_prompt_generation()
    multi_format_comparison()
    error_handling_demo()

    print("\n" + "=" * 60)
    print("Demo de integración con LLMs completado exitosamente!")
    print("Formatos disponibles: text, json, semantic")
    print("Casos de uso: documentación, RAG, análisis automático")
