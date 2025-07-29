import numpy as np
from typing import Dict, Any, List
from plot2llm.utils import generate_unified_key_insights, generate_unified_interpretation_hints

def analyze(ax, x_type=None, y_type=None) -> Dict[str, Any]:
    """
    Analiza un gráfico de líneas y devuelve información semántica completa.
    """
    # Información básica del eje
    section = {
        "plot_type": "line",
        "xlabel": str(ax.get_xlabel()),
        "ylabel": str(ax.get_ylabel()),
        "title": str(ax.get_title()),
        "x_lim": [float(x) for x in ax.get_xlim()],
        "y_lim": [float(y) for y in ax.get_ylim()],
        "x_range": [float(x) for x in ax.get_xlim()],  # Agregar x_range
        "y_range": [float(y) for y in ax.get_ylim()],  # Agregar y_range
        "has_grid": bool(any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())),
        "has_legend": bool(ax.get_legend() is not None)
    }
    
    # Añadir tipos de eje si se proporcionan
    if x_type:
        section["x_type"] = x_type
    if y_type:
        section["y_type"] = y_type
    
    # Extraer datos de las líneas
    lines_data = []
    all_x_data = []
    all_y_data = []
    curve_points = []
    
    for line in ax.lines:
        xdata = [float(x) for x in line.get_xdata()]
        ydata = [float(y) for y in line.get_ydata()]
        
        lines_data.append({
            "label": str(line.get_label()),
            "xdata": xdata,
            "ydata": ydata,
            "color": str(line.get_color()),
            "linestyle": str(line.get_linestyle()),
            "marker": str(line.get_marker())
        })
        
        # Agregar curve_points para esta línea
        curve_points.append({
            "x": xdata,
            "y": ydata,
            "type": "line",
            "color": str(line.get_color()),
            "style": str(line.get_linestyle()),
            "marker": str(line.get_marker()),
            "label": str(line.get_label())
        })
        
        all_x_data.extend(xdata)
        all_y_data.extend(ydata)
    
    section["lines"] = lines_data
    section["curve_points"] = curve_points
    
    # Análisis estadístico
    if all_y_data:
        y_array = np.array(all_y_data)
        x_array = np.array(all_x_data)
        y_clean = y_array[~np.isnan(y_array)]
        x_clean = x_array[~np.isnan(x_array)]
        
        # Detectar outliers usando IQR
        def detect_outliers(data):
            if len(data) < 4:
                return 0
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            if iqr == 0:
                return 0
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return int(np.sum((data < lower_bound) | (data > upper_bound)))
        
        y_outliers = detect_outliers(y_clean)
        
        stats = {
            "central_tendency": {
                "mean": float(np.nanmean(y_array)),
                "median": float(np.nanmedian(y_array)),
                "mode": None  # No calculamos mode para líneas
            },
            "variability": {
                "std": float(np.nanstd(y_array)),
                "variance": float(np.nanstd(y_array) ** 2),
                "range": {
                    "min": float(np.nanmin(y_array)),
                    "max": float(np.nanmax(y_array))
                }
            },
            "data_quality": {
                "total_points": int(len(all_y_data)),
                "missing_values": int(np.sum(np.isnan(y_array)))
            },
            "outliers": {
                "detected": bool(y_outliers > 0),
                "count": y_outliers
            }
        }
        
        # Análisis de tendencia
        slope = 0.0
        if len(y_clean) > 1 and len(x_clean) > 1:
            # Verificar si hay suficiente variación en los datos para evitar RankWarning
            x_std = np.std(x_clean)
            y_std = np.std(y_clean)
            
            if x_std > 1e-10 and y_std > 1e-10:  # Evitar datos constantes
                try:
                    # Calcular pendiente simple con datos limpios
                    slope = float(np.polyfit(x_clean, y_clean, 1)[0])
                    trend = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
                except (np.linalg.LinAlgError, ValueError):
                    # Si falla el polyfit, usar análisis simple
                    if len(y_clean) > 1:
                        first_val = y_clean[0]
                        last_val = y_clean[-1]
                        slope = (last_val - first_val) / (len(y_clean) - 1) if len(y_clean) > 1 else 0
                        trend = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
            else:
                trend = "unknown"
        else:
            trend = "unknown"
        
        # Análisis de patrones
        pattern_info = {
            "pattern_type": "linear_trend" if abs(slope) > 0.1 else "stable",
            "confidence_score": 0.9,
            "equation_estimate": f"y = {slope:.3f}x + b",
            "shape_characteristics": {
                "monotonicity": trend,
                "smoothness": "smooth",
                "symmetry": "unknown",
                "continuity": "continuous"
            }
        }
        
        section["stats"] = stats
        section["pattern"] = pattern_info
    
    # Generate LLM description and context
    section["llm_description"] = {
        "one_sentence_summary": f"This line plot shows {'an ' + trend + ' trend' if 'trend' in locals() else 'data relationships'} over the x-axis range.",
        "structured_analysis": {
            "what": "Line plot visualization",
            "when": "Time-series analysis" if x_type == "date" else "Sequential data analysis",
            "why": "Trend analysis and pattern recognition",
            "how": "Through continuous line representation of data points"
        },
        "key_insights": generate_unified_key_insights({
            "trend": trend if 'trend' in locals() and trend != "unknown" else None,
            "slope": slope if 'slope' in locals() else None,
            "pattern_confidence": pattern_info.get("confidence_score", 0.9) if 'pattern_info' in locals() else None
        })
    }
    
    section["llm_context"] = {
        "interpretation_hints": generate_unified_interpretation_hints({
            "trend_analysis": "Look for trends, slopes, and inflection points.",
            "direction_analysis": "Consider the overall direction and rate of change.",
            "pattern_recognition": "Identify any patterns or cycles in the data."
        }),
        "analysis_suggestions": [
            "Consider fitting a regression line for trend analysis.",
            "Look for seasonal patterns or periodic behavior.",
            "Analyze the rate of change at different points."
        ],
        "common_questions": [
            "Is there a clear trend in the data?",
            "Are there any significant changes in slope?",
            "What does the overall pattern suggest about the relationship?"
        ],
        "related_concepts": [
            "trend analysis",
            "regression",
            "time series",
            "slope calculation"
        ]
    }
    
    return section 