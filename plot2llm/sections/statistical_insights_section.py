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
            # Nueva estructura unificada
            if "central_tendency" in stats and "variability" in stats:
                # Estructura unificada
                converted_stats = {
                    "mean": stats["central_tendency"].get("mean"),
                    "median": stats["central_tendency"].get("median"),
                    "mode": stats["central_tendency"].get("mode"),
                    "std": stats["variability"].get("std"),
                    "variance": stats["variability"].get("variance"),
                    "min": stats["variability"]["range"].get("min") if "range" in stats["variability"] else None,
                    "max": stats["variability"]["range"].get("max") if "range" in stats["variability"] else None,
                    "data_points": stats["data_quality"].get("total_points") if "data_quality" in stats else None,
                    "missing_values": stats["data_quality"].get("missing_values") if "data_quality" in stats else None
                }
                
                # Agregar información específica del tipo de gráfico
                if "x_axis" in stats:
                    converted_stats["x_mean"] = stats["x_axis"].get("mean")
                    converted_stats["x_std"] = stats["x_axis"].get("std")
                    converted_stats["x_min"] = stats["x_axis"].get("min")
                    converted_stats["x_max"] = stats["x_axis"].get("max")
                
                if "categorical_analysis" in stats:
                    converted_stats["categorical_analysis"] = stats["categorical_analysis"]
                
                if "distribution_analysis" in stats:
                    converted_stats["distribution_analysis"] = stats["distribution_analysis"]
                
                axis_stats.append(converted_stats)
            else:
                # Estructura legacy - mantener compatibilidad
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
                "variance": primary_stats.get("variance"),
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
        
        # Add X-axis statistics if available
        if primary_stats.get("x_mean") is not None:
            insights["x_axis"] = {
                "mean": primary_stats.get("x_mean"),
                "std": primary_stats.get("x_std"),
                "min": primary_stats.get("x_min"),
                "max": primary_stats.get("x_max")
            }
        
        # Add categorical analysis if available (for bar plots)
        if primary_stats.get("categorical_analysis"):
            insights["categorical_analysis"] = primary_stats["categorical_analysis"]
        
        # Add distribution analysis if available (for histograms)
        if primary_stats.get("distribution_analysis"):
            insights["distribution_analysis"] = primary_stats["distribution_analysis"]
        
        return insights
    
    return {} 