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
            # Para scatter plots, shape_characteristics está dentro de pattern
            if "shape_characteristics" in pattern:
                shape_characteristics = pattern["shape_characteristics"]
                break
            # Para otros tipos de plots, puede estar en un campo shape separado
            elif "shape" in ax:
                shape_characteristics = ax["shape"]
                break
    
    return {
        "pattern_type": pattern_analysis_list[0].get("pattern_type") if pattern_analysis_list else None,
        "confidence_score": pattern_analysis_list[0].get("confidence_score") if pattern_analysis_list else None,
        "equation_estimate": pattern_analysis_list[0].get("equation_estimate") if pattern_analysis_list else None,
        "shape_characteristics": shape_characteristics,
    } 