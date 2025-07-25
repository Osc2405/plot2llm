import numpy as np
from typing import Dict, Any, List

def analyze(ax) -> Dict[str, Any]:
    """
    Analiza un gráfico de líneas y devuelve información semántica completa.
    """
    # Información básica del eje
    section = {
        "plot_type": "line",
        "x_label": str(ax.get_xlabel()),
        "y_label": str(ax.get_ylabel()),
        "title": str(ax.get_title()),
        "x_lim": [float(x) for x in ax.get_xlim()],
        "y_lim": [float(y) for y in ax.get_ylim()],
        "has_grid": bool(any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())),
        "has_legend": bool(ax.get_legend() is not None)
    }
    
    # Extraer datos de las líneas
    lines_data = []
    all_x_data = []
    all_y_data = []
    
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
        
        all_x_data.extend(xdata)
        all_y_data.extend(ydata)
    
    section["lines"] = lines_data
    
    # Análisis estadístico
    if all_y_data:
        y_array = np.array(all_y_data)
        x_array = np.array(all_x_data)
        
        # Estadísticas básicas
        stats = {
            "mean": float(np.nanmean(y_array)),
            "median": float(np.nanmedian(y_array)),
            "std": float(np.nanstd(y_array)),
            "min": float(np.nanmin(y_array)),
            "max": float(np.nanmax(y_array)),
            "range": float(np.nanmax(y_array) - np.nanmin(y_array)),
            "data_points": int(len(all_y_data))
        }
        
        # Análisis de tendencia
        slope = 0.0
        if len(y_array) > 1:
            # Calcular pendiente simple
            slope = float(np.polyfit(x_array, y_array, 1)[0])
            trend = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
        else:
            trend = "unknown"
        
        # Análisis de patrones
        pattern_info = {
            "pattern_type": "linear_trend" if abs(slope) > 0.1 else "stable",
            "trend_direction": trend,
            "slope": slope,
            "confidence_score": 0.8 if len(y_array) > 5 else 0.5
        }
        
        # Calcular equation_estimate
        if len(x_array) > 1 and len(y_array) > 1:
            try:
                coeffs = np.polyfit(x_array, y_array, 1)
                intercept = coeffs[1]
                pattern_info["equation_estimate"] = f"y = {slope:.2f}x + {intercept:.2f}"
            except Exception:
                pattern_info["equation_estimate"] = None
        else:
            pattern_info["equation_estimate"] = None
        
        section["statistics"] = stats
        section["pattern"] = pattern_info
    
    # Información del dominio
    domain_context = {
        "likely_domain": "time_series" if "time" in section["x_label"].lower() else "correlation",
        "purpose": "trend_analysis",
        "complexity_level": "medium" if len(lines_data) > 1 else "low"
    }
    section["domain_context"] = domain_context
    
    # Calcular shape_characteristics si hay datos
    if section.get("pattern") and all_y_data:
        y_array = np.array(all_y_data)
        x_array = np.array(all_x_data)
        
        if len(x_array) > 1 and len(y_array) > 1:
            # Análisis de características de forma
            shape_chars = {}
            
            # 1. Monotonicity
            diff_y = np.diff(y_array)
            increasing = np.sum(diff_y > 0)
            decreasing = np.sum(diff_y < 0)
            total_changes = len(diff_y)
            
            if increasing > 0.8 * total_changes:
                monotonicity = "increasing"
            elif decreasing > 0.8 * total_changes:
                monotonicity = "decreasing"
            else:
                monotonicity = "mixed"
            
            # 2. Smoothness
            if len(y_array) > 2:
                second_diff = np.diff(y_array, n=2)
                smoothness_var = np.var(second_diff) if len(second_diff) > 0 else 0
                if smoothness_var < 0.1:
                    smoothness = "smooth"
                elif smoothness_var < 1.0:
                    smoothness = "piecewise"
                else:
                    smoothness = "discrete"
            else:
                smoothness = "smooth"
            
            # 3. Symmetry
            y_center = len(y_array) // 2
            if len(y_array) > 4:
                left_half = y_array[:y_center]
                right_half = y_array[y_center:][:len(left_half)]
                if len(left_half) == len(right_half):
                    symmetry_corr = np.corrcoef(left_half, right_half[::-1])[0,1] if len(left_half) > 1 else 0
                    symmetry = "symmetric" if symmetry_corr > 0.8 else "asymmetric"
                else:
                    symmetry = "asymmetric"
            else:
                symmetry = "asymmetric"
            
            # 4. Continuity
            gaps = np.diff(x_array)
            avg_gap = np.mean(gaps) if len(gaps) > 0 else 0
            max_gap = np.max(gaps) if len(gaps) > 0 else 0
            
            if max_gap > 3 * avg_gap and avg_gap > 0:
                continuity = "discontinuous"
            else:
                continuity = "continuous"
            
            shape_chars = {
                "monotonicity": monotonicity,
                "smoothness": smoothness,
                "symmetry": symmetry,
                "continuity": continuity,
                "spread": float(np.max(y_array) - np.min(y_array)) if len(y_array) > 0 else None
            }
            
            section["pattern"]["shape_characteristics"] = shape_chars
    
    # Descripción para LLM
    if section.get("statistics"):
        trend_desc = section["pattern"]["trend_direction"]
        description = f"Line plot showing {trend_desc} trend with {section['statistics']['data_points']} data points"
    else:
        description = "Line plot visualization"
    
    llm_description = {
        "one_sentence_summary": description,
        "structured_analysis": {
            "what": f"Line chart with {len(lines_data)} series",
            "when": "Time-based analysis" if "time" in section["x_label"].lower() else "Correlation analysis",
            "why": "Trend identification and pattern recognition",
            "how": "Connected data points showing relationships over continuous variables"
        },
        "key_insights": [
            f"Data shows {section.get('pattern', {}).get('trend_direction', 'unknown')} trend",
            f"Range spans from {section.get('statistics', {}).get('min', 'N/A')} to {section.get('statistics', {}).get('max', 'N/A')}"
        ] if section.get("statistics") else []
    }
    section["llm_description"] = llm_description
    
    # Contexto para LLM
    llm_context = {
        "interpretation_hints": [
            "Analyze the slope and direction of the line(s)",
            "Look for patterns, trends, and outliers",
            "Consider the relationship between x and y variables"
        ],
        "analysis_suggestions": [
            "Calculate correlation coefficients",
            "Identify seasonal patterns if time-based",
            "Compare multiple series if present"
        ],
        "common_questions": [
            "What is the overall trend?",
            "Are there any significant changes in direction?",
            "How strong is the relationship between variables?"
        ],
        "related_concepts": [
            "trend analysis",
            "correlation",
            "time series" if "time" in section["x_label"].lower() else "regression"
        ]
    }
    section["llm_context"] = llm_context
    
    return section 