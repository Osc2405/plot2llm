def build_axes_section(semantic_analysis: dict, include_curve_points: bool = False) -> list:
    """
    Construye la sección axes para el output semántico.
    """
    axes = []
    for ax in semantic_analysis.get("axes", []):

        # Handle both modern (plot_type) and legacy (plot_types) formats
        plot_type = ax.get("plot_type")
        plot_types = ax.get("plot_types", [])
        
        # If we have the new format (plot_type), convert it to the expected format
        if plot_type and not plot_types:
            plot_types = [{"type": plot_type}]
        
        # Build the axes section
        axes_section = {
            "plot_types": plot_types,
            "xlabel": ax.get("xlabel") or ax.get("x_label", ""),
            "ylabel": ax.get("ylabel") or ax.get("y_label", ""),
            "title": ax.get("title", ""),
            "x_type": ax.get("x_type", "unknown"),
            "y_type": ax.get("y_type", "unknown"),
            "x_range": ax.get("x_range") or ax.get("x_lim", []),
            "y_range": ax.get("y_range") or ax.get("y_lim", []),
            "has_grid": ax.get("has_grid", False),
            "has_legend": ax.get("has_legend", False),
            "spine_visibility": ax.get("spine_visibility", {}),
            "tick_density": ax.get("tick_density", 0)
        }
        
        # Keep all stats fields - they are needed by statistical_insights section
        if "stats" in ax:
            stats = ax["stats"]
            # Keep all stats fields as they are needed by statistical_insights
            axes_section["stats"] = stats
        
        if include_curve_points:
            axes_section["curve_points"] = ax.get("curve_points", [])
            
        axes.append(axes_section)
    return axes 