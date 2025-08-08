"""
Ejemplos de casos de uso reales con plot2llm
Análisis financiero y de marketing con visualizaciones complejas
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import plot2llm


def financial_analysis_example():
    """
    Ejemplo de análisis financiero: Precios de acciones y distribución de retornos
    """
    print("=== ANÁLISIS FINANCIERO ===")

    # Simular datos de precios de acciones
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    prices = np.cumsum(np.random.randn(100) * 0.02) + 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Gráfico de precios
    ax1.plot(dates, prices, "b-", linewidth=2, label="Precio de Acción")
    ax1.set_title(
        "Evolución del Precio de Acciones (2023)", fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel("Precio ($)", fontsize=12)
    ax1.set_xlabel("Fecha", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Añadir línea de tendencia
    z = np.polyfit(range(len(prices)), prices, 1)
    p = np.poly1d(z)
    ax1.plot(dates, p(range(len(prices))), "r--", alpha=0.8, label="Tendencia")

    # Histograma de retornos
    returns = np.diff(prices) / prices[:-1] * 100
    ax2.hist(returns, bins=20, alpha=0.7, color="green", edgecolor="black")
    ax2.set_title("Distribución de Retornos Diarios", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Retorno (%)", fontsize=12)
    ax2.set_ylabel("Frecuencia", fontsize=12)
    ax2.axvline(
        np.mean(returns),
        color="red",
        linestyle="--",
        label=f"Media: {np.mean(returns):.2f}%",
    )
    ax2.legend()

    plt.tight_layout()

    # Convertir con plot2llm
    result = plot2llm.convert(fig, format="semantic", detail_level="high")
    print("Análisis semántico del gráfico financiero:")
    print(result)
    print("\n" + "=" * 50 + "\n")

    return fig


def marketing_analysis_example():
    """
    Ejemplo de análisis de marketing: Conversiones por canal y ROI
    """
    print("=== ANÁLISIS DE MARKETING ===")

    # Simular datos de campaña de marketing
    categories = ["Email", "Social Media", "PPC", "Organic", "Direct"]
    conversions = [120, 85, 200, 150, 75]
    costs = [500, 300, 800, 200, 100]

    # Calcular ROI
    roi = [(conv * 50 - cost) / cost * 100 for conv, cost in zip(conversions, costs)]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Gráfico 1: Conversiones por canal
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
    bars1 = ax1.bar(categories, conversions, color=colors, alpha=0.8)
    ax1.set_title("Conversiones por Canal de Marketing", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Conversiones", fontsize=12)
    ax1.set_xlabel("Canal", fontsize=12)

    # Añadir valores en las barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Gráfico 2: ROI por canal
    bars2 = ax2.bar(categories, roi, color=colors, alpha=0.8)
    ax2.set_title("ROI por Canal de Marketing", fontsize=14, fontweight="bold")
    ax2.set_ylabel("ROI (%)", fontsize=12)
    ax2.set_xlabel("Canal", fontsize=12)
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontweight="bold",
        )

    # Gráfico 3: Dispersión costo vs conversión
    _ = ax3.scatter(
        costs, conversions, s=100, alpha=0.7, c=range(len(costs)), cmap="viridis"
    )
    ax3.set_xlabel("Costo ($)", fontsize=12)
    ax3.set_ylabel("Conversiones", fontsize=12)
    ax3.set_title("Costo vs Conversiones", fontsize=14, fontweight="bold")

    # Añadir etiquetas
    for i, cat in enumerate(categories):
        ax3.annotate(
            cat,
            (costs[i], conversions[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    # Gráfico 4: Pie chart de distribución de costos
    ax4.pie(costs, labels=categories, autopct="%1.1f%%", colors=colors)
    ax4.set_title("Distribución de Costos por Canal", fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Convertir con plot2llm
    result = plot2llm.convert(fig, format="semantic", detail_level="high")
    print("Análisis semántico del gráfico de marketing:")
    print(result)
    print("\n" + "=" * 50 + "\n")

    return fig


def customer_segmentation_example():
    """
    Ejemplo de segmentación de clientes con análisis demográfico
    """
    print("=== ANÁLISIS DE SEGMENTACIÓN DE CLIENTES ===")

    # Simular datos de clientes
    np.random.seed(42)
    n_customers = 1000

    # Generar datos demográficos
    ages = np.random.normal(35, 12, n_customers)
    ages = np.clip(ages, 18, 70)

    income = np.random.lognormal(10.5, 0.4, n_customers)
    income = np.clip(income, 20000, 200000)

    satisfaction = np.random.uniform(1, 10, n_customers)

    # Crear segmentos
    segments = []
    for age, inc, _sat in zip(ages, income, satisfaction):
        if age < 30 and inc < 50000:
            segments.append("Joven Básico")
        elif age < 30 and inc >= 50000:
            segments.append("Joven Premium")
        elif age >= 30 and inc < 80000:
            segments.append("Adulto Básico")
        else:
            segments.append("Adulto Premium")

    data = pd.DataFrame(
        {
            "edad": ages,
            "ingresos": income,
            "satisfaccion": satisfaction,
            "segmento": segments,
        }
    )

    fig = plt.figure(figsize=(16, 10))

    # Subplot 1: Distribución de edad por segmento
    plt.subplot(2, 3, 1)
    for segment in data["segmento"].unique():
        segment_data = data[data["segmento"] == segment]
        plt.hist(segment_data["edad"], alpha=0.6, label=segment, bins=20)
    plt.title("Distribución de Edad por Segmento", fontweight="bold")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    plt.legend()

    # Subplot 2: Ingresos vs Satisfacción
    plt.subplot(2, 3, 2)
    colors = ["red", "blue", "green", "orange"]
    for i, segment in enumerate(data["segmento"].unique()):
        segment_data = data[data["segmento"] == segment]
        plt.scatter(
            segment_data["ingresos"],
            segment_data["satisfaccion"],
            alpha=0.6,
            label=segment,
            color=colors[i],
        )
    plt.title("Ingresos vs Satisfacción por Segmento", fontweight="bold")
    plt.xlabel("Ingresos ($)")
    plt.ylabel("Satisfacción")
    plt.legend()

    # Subplot 3: Boxplot de ingresos por segmento
    plt.subplot(2, 3, 3)
    data.boxplot(column="ingresos", by="segmento", ax=plt.gca())
    plt.title("Distribución de Ingresos por Segmento", fontweight="bold")
    plt.suptitle("")

    # Subplot 4: Heatmap de correlaciones
    plt.subplot(2, 3, 4)
    corr_matrix = data[["edad", "ingresos", "satisfaccion"]].corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Matriz de Correlaciones", fontweight="bold")

    # Subplot 5: Violin plot de satisfacción
    plt.subplot(2, 3, 5)
    sns.violinplot(data=data, x="segmento", y="satisfaccion")
    plt.title("Satisfacción por Segmento", fontweight="bold")
    plt.xticks(rotation=45)

    # Subplot 6: Distribución de segmentos
    plt.subplot(2, 3, 6)
    segment_counts = data["segmento"].value_counts()
    plt.pie(segment_counts.values, labels=segment_counts.index, autopct="%1.1f%%")
    plt.title("Distribución de Segmentos", fontweight="bold")

    plt.tight_layout()

    # Convertir con plot2llm
    result = plot2llm.convert(fig, format="semantic", detail_level="high")
    print("Análisis semántico del gráfico de segmentación:")
    print(result)
    print("\n" + "=" * 50 + "\n")

    return fig


if __name__ == "__main__":
    print("EJEMPLOS DE CASOS DE USO REALES CON PLOT2LLM")
    print("=" * 60)

    # Ejecutar ejemplos
    financial_analysis_example()
    marketing_analysis_example()
    customer_segmentation_example()

    print("Todos los ejemplos completados exitosamente!")
