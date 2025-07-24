def build_statistical_insights_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección statistical_insights para el output semántico.
    """
    axes = semantic_analysis.get("axes", [])
    statistical_insights_list = [ax.get("stats", {}) for ax in axes]
    return statistical_insights_list[0] if statistical_insights_list else {
        "trend": None,
        "distribution": None,
        "correlations": [],
        "key_statistics": None,
    } 