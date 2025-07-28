def build_statistical_insights_section(semantic_analysis: dict) -> dict:
    """
    Construye la sección statistical_insights para el output semántico.
    """
    axes = semantic_analysis.get("axes", [])
    statistics = semantic_analysis.get("statistics", {})
    
    # Buscar estadísticas en los ejes individuales
    axis_stats = []
    for ax in axes:
        # Buscar en diferentes campos donde pueden estar las estadísticas
        stats = ax.get("stats", {}) or ax.get("statistics", {})
        if stats:
            # Handle scatter plot statistics format (x_stats, y_stats)
            if "x_stats" in stats and "y_stats" in stats:
                # Convert scatter format to standard format
                x_stats = stats["x_stats"]
                y_stats = stats["y_stats"]
                
                # Use Y stats as primary (since Y is typically the dependent variable)
                converted_stats = {
                    "mean": y_stats.get("mean"),
                    "median": y_stats.get("median"),
                    "std": y_stats.get("std"),
                    "min": y_stats.get("min"),
                    "max": y_stats.get("max"),
                    "data_points": stats.get("data_points"),
                    "x_mean": x_stats.get("mean"),
                    "x_std": x_stats.get("std"),
                    "x_min": x_stats.get("min"),
                    "x_max": x_stats.get("max"),
                    "correlation": stats.get("correlation", 0.0),
                    "correlation_strength": stats.get("correlation_strength", "unknown"),
                    "correlation_direction": stats.get("correlation_direction", "unknown")
                }
                axis_stats.append(converted_stats)
            else:
                # Standard format
                axis_stats.append(stats)
    
    # Si no hay estadísticas en los ejes, buscar en el campo statistics principal
    if not axis_stats and "per_axis" in statistics:
        axis_stats = statistics["per_axis"]
    
    # Si aún no hay estadísticas, usar las estadísticas globales
    if not axis_stats and "global" in statistics:
        global_stats = statistics["global"]
        axis_stats = [{
            "mean": global_stats.get("mean"),
            "median": global_stats.get("median"),
            "std": global_stats.get("std"),
            "min": global_stats.get("min"),
            "max": global_stats.get("max"),
            "data_points": global_stats.get("data_points")
        }]
    
    # Construir insights estadísticos
    if axis_stats:
        primary_stats = axis_stats[0]  # Usar las estadísticas del primer eje
        
        insights = {
            "central_tendency": {
                "mean": primary_stats.get("mean"),
                "median": primary_stats.get("median"),
                "mode": primary_stats.get("mode")
            },
            "variability": {
                "standard_deviation": primary_stats.get("std"),
                "variance": primary_stats.get("std", 0) ** 2 if primary_stats.get("std") else None,
                "range": {
                    "min": primary_stats.get("min"),
                    "max": primary_stats.get("max")
                }
            },
            "data_quality": {
                "total_points": primary_stats.get("data_points"),
                "missing_values": primary_stats.get("missing_values", 0)
            },
            "distribution": {
                "skewness": primary_stats.get("skewness"),
                "kurtosis": primary_stats.get("kurtosis")
            },
            "correlations": primary_stats.get("correlations", []),
            "outliers": primary_stats.get("outliers", {})
        }
        
        # Add correlation information if available (for scatter plots)
        if primary_stats.get("correlation") is not None:
            insights["correlations"] = [{
                "type": "pearson",
                "value": primary_stats.get("correlation"),
                "strength": primary_stats.get("correlation_strength"),
                "direction": primary_stats.get("correlation_direction")
            }]
        
        # Also check for correlation information in pattern field (scatter plots)
        for ax in axes:
            if ax.get("pattern") and isinstance(ax["pattern"], dict):
                pattern = ax["pattern"]
                if pattern.get("correlation") is not None:
                    insights["correlations"] = [{
                        "type": "pearson",
                        "value": pattern.get("correlation"),
                        "strength": pattern.get("correlation_strength"),
                        "direction": pattern.get("correlation_direction")
                    }]
                    break
        
        # Add X-axis statistics if available (for scatter plots)
        if primary_stats.get("x_mean") is not None:
            insights["x_axis"] = {
                "mean": primary_stats.get("x_mean"),
                "std": primary_stats.get("x_std"),
                "range": {
                    "min": primary_stats.get("x_min"),
                    "max": primary_stats.get("x_max")
                }
            }
        
        # Limpiar valores None
        insights = {k: v for k, v in insights.items() if v is not None}
        
        return insights
    else:
        # Retornar estructura vacía si no hay estadísticas
        return {
            "trend": None,
            "distribution": None,
            "correlations": [],
            "key_statistics": None,
        } 