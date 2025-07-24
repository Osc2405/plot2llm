def build_pattern_analysis_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección pattern_analysis para el output semántico.
    """
    axes = semantic_analysis.get("axes", [])
    pattern_analysis_list = [ax.get("pattern", {}) for ax in axes]
    shape_characteristics_list = [ax.get("shape", {}) for ax in axes]
    return {
        "pattern_type": pattern_analysis_list[0].get("pattern_type") if pattern_analysis_list else None,
        "confidence_score": pattern_analysis_list[0].get("confidence_score") if pattern_analysis_list else None,
        "equation_estimate": pattern_analysis_list[0].get("equation_estimate") if pattern_analysis_list else None,
        "shape_characteristics": shape_characteristics_list[0] if shape_characteristics_list else None,
    } 