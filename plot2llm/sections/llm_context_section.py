from plot2llm.utils import generate_unified_interpretation_hints


def build_llm_context_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección llm_context para el output semántico.
    """
    axes = semantic_analysis.get("axes", [])

    # Buscar contexto LLM específico en los ejes (generado por los analizadores)
    for ax in axes:
        if "llm_context" in ax and ax["llm_context"]:
            return ax["llm_context"]

    # Si no hay contexto específico, buscar en el análisis principal
    if "llm_context" in semantic_analysis and semantic_analysis["llm_context"]:
        return semantic_analysis["llm_context"]

    # Fallback: construir contexto basado en tipos de gráfico
    plot_types = set()
    for ax in axes:
        for pt in ax.get("plot_types", []):
            if pt.get("type"):
                plot_types.add(pt["type"])

    # Generate unified interpretation hints
    hints_data = {}

    if plot_types:
        if "line" in plot_types:
            hints_data["trend_analysis"] = (
                "Look for trends, slopes, and inflection points."
            )
            hints_data["direction_analysis"] = (
                "Consider the overall direction and rate of change."
            )
            hints_data["pattern_recognition"] = (
                "Identify any patterns or cycles in the data."
            )
        if "scatter" in plot_types:
            hints_data["cluster_analysis"] = (
                "Check for clusters, outliers, and correlation between variables."
            )
            hints_data["pattern_recognition"] = (
                "Look for patterns in the distribution of points."
            )
            hints_data["correlation_analysis"] = (
                "Consider the strength and direction of any relationship."
            )
        if "histogram" in plot_types:
            hints_data["shape_analysis"] = "Observe the distribution shape and spread."
            hints_data["peak_analysis"] = (
                "Look for peaks, valleys, and overall symmetry."
            )
            hints_data["statistical_analysis"] = (
                "Consider the skewness and kurtosis of the distribution."
            )
        if "bar" in plot_types:
            hints_data["categorical_comparison"] = (
                "Compare the heights of the bars for categorical differences."
            )
            hints_data["ranking_analysis"] = (
                "Look for the largest and smallest categories."
            )
            hints_data["distribution_analysis"] = (
                "Consider the overall distribution of values across categories."
            )

    # Si no se encontraron hints específicos, usar genéricos
    if not hints_data:
        hints_data["general_analysis"] = (
            "Interpret the axes, labels, and data points to understand the visualization."
        )

    hints = generate_unified_interpretation_hints(hints_data)

    # Generate suggestions and questions (keeping original format for now)
    suggestions = []
    questions = []
    concepts = []

    if plot_types:
        if "line" in plot_types:
            suggestions.append(
                "Consider fitting a regression or analyzing periodicity."
            )
            questions.append("Is there a clear trend or periodic pattern in the data?")
            concepts.extend(["trend analysis", "regression", "time series"])
        if "scatter" in plot_types:
            suggestions.append(
                "Try calculating the correlation coefficient or clustering."
            )
            questions.append("Are the variables correlated? Are there any outliers?")
            concepts.extend(["correlation", "outlier detection", "clustering"])
        if "histogram" in plot_types:
            suggestions.append(
                "Estimate skewness, kurtosis, and check for multimodality."
            )
            questions.append("Is the distribution normal, skewed, or multimodal?")
            concepts.extend(["distribution", "skewness", "kurtosis"])
        if "bar" in plot_types:
            suggestions.append("Look for the largest and smallest categories.")
            questions.append("Which category has the highest/lowest value?")
            concepts.extend(["categorical comparison", "ranking"])

    # Si no se encontraron suggestions específicos, usar genéricos
    if not suggestions:
        suggestions.append("Explore summary statistics and relationships in the data.")
    if not questions:
        questions.append("What does this plot reveal about the data?")
    if not concepts:
        concepts.append("data visualization")

    return {
        "interpretation_hints": hints,
        "analysis_suggestions": suggestions,
        "common_questions": questions,
        "related_concepts": list(set(concepts)),
    }
