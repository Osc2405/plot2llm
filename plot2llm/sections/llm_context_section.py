def build_llm_context_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección llm_context para el output semántico.
    """
    axes = semantic_analysis.get("axes", [])
    plot_types = set()
    for ax in axes:
        for pt in ax.get("plot_types", []):
            if pt.get("type"):
                plot_types.add(pt["type"])
    hints = []
    suggestions = []
    questions = []
    concepts = []
    if plot_types:
        if "line" in plot_types:
            hints.append("Look for trends, slopes, and inflection points.")
            suggestions.append("Consider fitting a regression or analyzing periodicity.")
            questions.append("Is there a clear trend or periodic pattern in the data?")
            concepts.extend(["trend analysis", "regression", "time series"])
        if "scatter" in plot_types:
            hints.append("Check for clusters, outliers, and correlation between variables.")
            suggestions.append("Try calculating the correlation coefficient or clustering.")
            questions.append("Are the variables correlated? Are there any outliers?")
            concepts.extend(["correlation", "outlier detection", "clustering"])
        if "histogram" in plot_types:
            hints.append("Observe the distribution shape and spread.")
            suggestions.append("Estimate skewness, kurtosis, and check for multimodality.")
            questions.append("Is the distribution normal, skewed, or multimodal?")
            concepts.extend(["distribution", "skewness", "kurtosis"])
        if "bar" in plot_types:
            hints.append("Compare the heights of the bars for categorical differences.")
            suggestions.append("Look for the largest and smallest categories.")
            questions.append("Which category has the highest/lowest value?")
            concepts.extend(["categorical comparison", "ranking"])
    if not hints:
        hints.append("Interpret the axes, labels, and data points to understand the visualization.")
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