def build_data_summary_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección data_summary para el output semántico.
    """
    data_info = semantic_analysis.get("data_info", {})
    axes = semantic_analysis.get("axes", [])
    x_type = axes[0].get("x_type") if axes else None
    y_type = axes[0].get("y_type") if axes else None
    x_range = axes[0].get("x_range", [None, None]) if axes else None
    y_range = axes[0].get("y_range", [None, None]) if axes else None
    data_summary = {
        "total_data_points": data_info.get("data_points", 0),
        "data_ranges": {
            "x": {"min": x_range[0], "max": x_range[1], "type": x_type} if x_range else None,
            "y": {"min": y_range[0], "max": y_range[1], "type": y_type} if y_range else None,
        },
        "missing_values": None,
        "x_type": x_type,
        "y_type": y_type,
    }
    return data_summary 