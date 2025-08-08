def build_pattern_analysis_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección pattern_analysis para el output semántico.
    """
    axes = semantic_analysis.get("axes", [])
    pattern_analysis_list = [ax.get("pattern", {}) for ax in axes]

    # Buscar shape_characteristics en diferentes ubicaciones
    shape_characteristics = None
    for ax in axes:
        pattern = ax.get("pattern", {})
        if pattern and isinstance(pattern, dict):
            # Para todos los tipos de plots, shape_characteristics está dentro de pattern
            if "shape_characteristics" in pattern:
                shape_characteristics = pattern["shape_characteristics"]
                break
            # Para otros tipos de plots, puede estar en un campo shape separado
            elif "shape" in ax:
                shape_characteristics = ax["shape"]
                break

    # Construir pattern_analysis con estructura unificada
    if pattern_analysis_list and pattern_analysis_list[0]:
        primary_pattern = pattern_analysis_list[0]

        pattern_analysis = {
            "pattern_type": primary_pattern.get("pattern_type"),
            "confidence_score": primary_pattern.get("confidence_score"),
            "equation_estimate": primary_pattern.get("equation_estimate"),
            "shape_characteristics": shape_characteristics,
        }

        # Agregar características específicas del tipo de gráfico
        if "correlation" in primary_pattern:
            pattern_analysis["correlation"] = primary_pattern.get("correlation")
            pattern_analysis["correlation_strength"] = primary_pattern.get(
                "correlation_strength"
            )
            pattern_analysis["correlation_direction"] = primary_pattern.get(
                "correlation_direction"
            )

        if "distribution_characteristics" in primary_pattern:
            pattern_analysis["distribution_characteristics"] = primary_pattern[
                "distribution_characteristics"
            ]

        return pattern_analysis

    return {
        "pattern_type": None,
        "confidence_score": None,
        "equation_estimate": None,
        "shape_characteristics": shape_characteristics,
    }
