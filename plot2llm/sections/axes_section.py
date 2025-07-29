def build_axes_section(semantic_analysis: dict, include_curve_points: bool = False) -> list:
    """
    Construye la sección axes para el output semántico.
    """
    axes = []
    for ax in semantic_analysis.get("axes", []):

        # Handle both modern and legacy axis formats
        # Handle both modern (plot_type) and legacy (plot_types) formats
        plot_type = ax.get("plot_type")
        plot_types = ax.get("plot_types", [])
        
        # If we have the new format (plot_type), convert it to the expected format
        if plot_type and not plot_types:
            plot_types = [{"type": plot_type}]
        
        # Start with the original axis data to preserve all fields
        axis_entry = dict(ax)
        
        # Update/add specific fields that need to be standardized
        axis_entry.update({
            "title": ax.get("title", ""),
            "xlabel": ax.get("xlabel") or ax.get("x_label", ""),
            "ylabel": ax.get("ylabel") or ax.get("y_label", ""),
            "plot_types": plot_types,
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
        })
        
        if include_curve_points:
            axis_entry["curve_points"] = ax.get("curve_points", [])
            
        axes.append(axis_entry)
    return axes 