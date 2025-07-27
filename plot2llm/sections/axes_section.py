def build_axes_section(semantic_analysis: dict, include_curve_points: bool = False) -> list:
    """
    Construye la sección axes para el output semántico.
    """
    axes = []
    for ax in semantic_analysis.get("axes", []):
        # Handle both modern and legacy axis formats
        axis_entry = {
            "title": ax.get("title", ""),
            "xlabel": ax.get("xlabel") or ax.get("x_label", ""),
            "ylabel": ax.get("ylabel") or ax.get("y_label", ""),
            "plot_types": ax.get("plot_types", []),
            "x_type": ax.get("x_type", "unknown"),
            "y_type": ax.get("y_type", "unknown"),
            "has_grid": ax.get("has_grid", False),
            "has_legend": ax.get("has_legend", False),
            "x_range": ax.get("x_range") or ax.get("x_lim"),
            "y_range": ax.get("y_range") or ax.get("y_lim"),
            "spine_visibility": ax.get("spine_visibility"),
            "tick_density": ax.get("tick_density"),
            "pattern": ax.get("pattern"),
            "shape": ax.get("shape"),
            "domain_context": ax.get("domain_context"),
            "stats": ax.get("stats") or ax.get("statistics"),
        }
        if include_curve_points:
            axis_entry["curve_points"] = ax.get("curve_points", [])
        axes.append(axis_entry)
    return axes 