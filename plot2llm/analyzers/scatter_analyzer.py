import numpy as np
from typing import Dict, Any, List

def analyze(ax, x_type=None, y_type=None) -> Dict[str, Any]:
    """
    Analiza un scatter plot y devuelve información semántica completa.
    """
    # Información básica del eje
    section = {
        "plot_type": "scatter",
        "x_label": str(ax.get_xlabel()),
        "y_label": str(ax.get_ylabel()),
        "title": str(ax.get_title()),
        "x_lim": [float(x) for x in ax.get_xlim()],
        "y_lim": [float(y) for y in ax.get_ylim()],
        "has_grid": bool(any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())),
        "has_legend": bool(ax.get_legend() is not None),
    }
    
    # Añadir tipos de eje si se proporcionan
    if x_type:
        section["x_type"] = x_type
    if y_type:
        section["y_type"] = y_type
    
    # Extraer datos de las colecciones (scatter points)
    collections_data = []
    all_x_data = []
    all_y_data = []
    
    for collection in ax.collections:
        if hasattr(collection, "get_offsets"):
            offsets = collection.get_offsets()
            if len(offsets) > 0:
                x_points = [float(x) for x in offsets[:, 0]]
                y_points = [float(y) for y in offsets[:, 1]]
                
                collections_data.append({
                    "label": str(getattr(collection, "get_label", lambda: "scatter_data")()),
                    "x_data": x_points,
                    "y_data": y_points,
                    "n_points": int(len(x_points))
                })
                
                all_x_data.extend(x_points)
                all_y_data.extend(y_points)
    
    section["collections"] = collections_data
    
    # Análisis estadístico
    if all_x_data and all_y_data:
        x_array = np.array(all_x_data)
        y_array = np.array(all_y_data)
        
        # Estadísticas básicas
        stats = {
            "x_stats": {
                "mean": float(np.nanmean(x_array)),
                "std": float(np.nanstd(x_array)),
                "min": float(np.nanmin(x_array)),
                "max": float(np.nanmax(x_array))
            },
            "y_stats": {
                "mean": float(np.nanmean(y_array)),
                "std": float(np.nanstd(y_array)),
                "min": float(np.nanmin(y_array)),
                "max": float(np.nanmax(y_array))
            },
            "data_points": int(len(all_x_data))
        }
        
        # Análisis de correlación
        correlation = 0.0
        if len(x_array) > 1 and len(y_array) > 1:
            corr_matrix = np.corrcoef(x_array, y_array)
            correlation = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
        
        # Análisis de distribución
        x_spread = float(np.nanmax(x_array) - np.nanmin(x_array))
        y_spread = float(np.nanmax(y_array) - np.nanmin(y_array))
        
        # Detectar outliers usando IQR
        def detect_outliers(data):
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return int(np.sum((data < lower_bound) | (data > upper_bound)))
        
        x_outliers = detect_outliers(x_array) if len(x_array) > 4 else 0
        y_outliers = detect_outliers(y_array) if len(y_array) > 4 else 0
        
        # Análisis de patrones
        pattern_info = {
            "correlation": correlation,
            "correlation_strength": (
                "strong" if abs(correlation) > 0.7 else
                "moderate" if abs(correlation) > 0.3 else
                "weak"
            ),
            "correlation_direction": (
                "positive" if correlation > 0.1 else
                "negative" if correlation < -0.1 else
                "no_correlation"
            ),
            "distribution_type": "scattered",
            "x_spread": x_spread,
            "y_spread": y_spread,
            "outliers": {
                "x_outliers": x_outliers,
                "y_outliers": y_outliers
            }
        }
        
        # Agregar pattern_type y confidence_score para compatibilidad con pattern_analysis
        if abs(correlation) > 0.7:
            pattern_info["pattern_type"] = "strong_correlation"
            pattern_info["confidence_score"] = 0.9
        elif abs(correlation) > 0.3:
            pattern_info["pattern_type"] = "moderate_correlation" 
            pattern_info["confidence_score"] = 0.7
        else:
            pattern_info["pattern_type"] = "weak_correlation"
            pattern_info["confidence_score"] = 0.4
            
        # Agregar equation_estimate si hay correlación fuerte
        if abs(correlation) > 0.5 and len(x_array) > 1 and len(y_array) > 1:
            try:
                # Filtrar NaN values antes del polyfit
                valid_mask = ~(np.isnan(x_array) | np.isnan(y_array))
                x_clean = x_array[valid_mask]
                y_clean = y_array[valid_mask]
                
                if len(x_clean) > 1 and len(y_clean) > 1:
                    # Verificar si hay suficiente variación en los datos para evitar RankWarning
                    x_std = np.std(x_clean)
                    y_std = np.std(y_clean)
                    
                    if x_std > 1e-10 and y_std > 1e-10:  # Evitar datos constantes
                        coeffs = np.polyfit(x_clean, y_clean, 1)
                        slope, intercept = coeffs[0], coeffs[1]
                        pattern_info["equation_estimate"] = f"y = {slope:.2f}x + {intercept:.2f}"
                    else:
                        pattern_info["equation_estimate"] = None
                else:
                    pattern_info["equation_estimate"] = None
            except (np.linalg.LinAlgError, ValueError):
                pattern_info["equation_estimate"] = None
        else:
            pattern_info["equation_estimate"] = None
            
        # Análisis de características de forma para scatter plots
        if len(x_array) > 2 and len(y_array) > 2:
            # Filtrar NaN values para el análisis de forma
            valid_mask = ~(np.isnan(x_array) | np.isnan(y_array))
            x_clean = x_array[valid_mask]
            y_clean = y_array[valid_mask]
            
            if len(x_clean) > 2 and len(y_clean) > 2:
                # Ordenar por x para análisis de monotonicity y smoothness
                sorted_indices = np.argsort(x_clean)
                x_sorted = x_clean[sorted_indices]
                y_sorted = y_clean[sorted_indices]
                
                shape_chars = {}
                
                # 1. Monotonicity (basado en la tendencia general)
                if abs(correlation) > 0.3:
                    if correlation > 0:
                        monotonicity = "increasing"
                    else:
                        monotonicity = "decreasing"
                else:
                    # Analizar cambios locales si no hay correlación clara
                    diff_y = np.diff(y_sorted)
                    increasing = np.sum(diff_y > 0)
                    decreasing = np.sum(diff_y < 0)
                    total_changes = len(diff_y)
                    
                    if increasing > 0.6 * total_changes:
                        monotonicity = "increasing"
                    elif decreasing > 0.6 * total_changes:
                        monotonicity = "decreasing"
                    else:
                        monotonicity = "mixed"
                
                # 2. Smoothness (basado en la variabilidad de los residuos)
                if abs(correlation) > 0.3:
                    try:
                        # Verificar si hay suficiente variación en los datos para evitar RankWarning
                        x_std = np.std(x_sorted)
                        y_std = np.std(y_sorted)
                        
                        if x_std > 1e-10 and y_std > 1e-10:  # Evitar datos constantes
                            coeffs = np.polyfit(x_sorted, y_sorted, 1)
                            y_pred = np.polyval(coeffs, x_sorted)
                            residuals = y_sorted - y_pred
                            residual_var = np.var(residuals)
                            
                            if residual_var < np.var(y_sorted) * 0.1:
                                smoothness = "smooth"
                            elif residual_var < np.var(y_sorted) * 0.5:
                                smoothness = "piecewise"
                            else:
                                smoothness = "discrete"
                        else:
                            smoothness = "discrete"
                    except (np.linalg.LinAlgError, ValueError):
                        smoothness = "discrete"
                else:
                    smoothness = "discrete"
                
                # 3. Symmetry (distribución de puntos respecto al centro)
                x_center = np.median(x_clean)
                y_center = np.median(y_clean)
                
                # Dividir en cuadrantes y analizar distribución
                q1 = np.sum((x_clean < x_center) & (y_clean < y_center))
                q2 = np.sum((x_clean >= x_center) & (y_clean < y_center))
                q3 = np.sum((x_clean >= x_center) & (y_clean >= y_center))
                q4 = np.sum((x_clean < x_center) & (y_clean >= y_center))
                
                # Simétrica si q1≈q3 y q2≈q4
                total_points = len(x_clean)
                symmetry_score = abs((q1 + q3) - (q2 + q4)) / total_points
                symmetry = "symmetric" if symmetry_score < 0.2 else "asymmetric"
                
                # 4. Continuity (basado en la densidad de distribución)
                # Para scatter plots, analizamos si hay gaps significativos
                x_gaps = np.diff(np.sort(x_clean))
                y_gaps = np.diff(np.sort(y_clean))
                
                x_avg_gap = np.mean(x_gaps) if len(x_gaps) > 0 else 0
                y_avg_gap = np.mean(y_gaps) if len(y_gaps) > 0 else 0
                x_max_gap = np.max(x_gaps) if len(x_gaps) > 0 else 0
                y_max_gap = np.max(y_gaps) if len(y_gaps) > 0 else 0
                
                # Considerar discontinuo si hay gaps muy grandes
                if (x_max_gap > 3 * x_avg_gap and x_avg_gap > 0) or (y_max_gap > 3 * y_avg_gap and y_avg_gap > 0):
                    continuity = "discontinuous"
                else:
                    continuity = "continuous"
                
                shape_chars = {
                    "monotonicity": monotonicity,
                    "smoothness": smoothness,
                    "symmetry": symmetry,
                    "continuity": continuity,
                    "correlation": correlation,
                    "correlation_strength": pattern_info["correlation_strength"],
                    "correlation_direction": pattern_info["correlation_direction"],
                    "x_spread": x_spread,
                    "y_spread": y_spread,
                    "outliers": pattern_info["outliers"]
                }
                
                pattern_info["shape_characteristics"] = shape_chars
        
        section["statistics"] = stats
        section["pattern"] = pattern_info
    
    # Información del dominio
    domain_context = {
        "likely_domain": "correlation_analysis",
        "purpose": "relationship_exploration",
        "complexity_level": "medium" if len(collections_data) > 1 else "low",
        "analysis_type": "bivariate_analysis"
    }
    section["domain_context"] = domain_context
    
    # Descripción para LLM
    if section.get("statistics"):
        corr_desc = section["pattern"]["correlation_strength"]
        corr_dir = section["pattern"]["correlation_direction"]
        description = f"Scatter plot showing {corr_desc} {corr_dir} correlation with {section['statistics']['data_points']} data points"
    else:
        description = "Scatter plot visualization"
    
    llm_description = {
        "one_sentence_summary": description,
        "structured_analysis": {
            "what": f"Scatter plot with {len(collections_data)} data series",
            "when": "Comparative analysis between two continuous variables",
            "why": "To identify relationships, correlations, and patterns between variables",
            "how": "Individual data points plotted on x-y coordinate system"
        },
        "key_insights": [
            f"Correlation strength: {section.get('pattern', {}).get('correlation_strength', 'unknown')}",
            f"Correlation direction: {section.get('pattern', {}).get('correlation_direction', 'unknown')}",
            f"Data distribution spans X: {section.get('pattern', {}).get('x_spread', 'N/A')}, Y: {section.get('pattern', {}).get('y_spread', 'N/A')}"
        ] if section.get("statistics") else []
    }
    section["llm_description"] = llm_description
    
    # Contexto para LLM
    llm_context = {
        "interpretation_hints": [
            "Look for clustering patterns and outliers",
            "Assess the strength and direction of correlation",
            "Consider the distribution and spread of data points"
        ],
        "analysis_suggestions": [
            "Calculate Pearson correlation coefficient",
            "Identify potential outliers and their impact",
            "Consider logarithmic or other transformations if needed"
        ],
        "common_questions": [
            "What is the relationship between the two variables?",
            "Are there any outliers or clusters?",
            "How strong is the correlation?"
        ],
        "related_concepts": [
            "correlation analysis",
            "bivariate statistics",
            "outlier detection",
            "regression analysis"
        ]
    }
    section["llm_context"] = llm_context
    
    return section 