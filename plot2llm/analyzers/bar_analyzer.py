import numpy as np
from typing import Dict, Any, List

def analyze(ax, x_type=None, y_type=None) -> Dict[str, Any]:
    """
    Analiza un gráfico de barras y devuelve información semántica completa.
    """
    # Información básica del eje
    section = {
        "plot_type": "bar",
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
    
    # Extraer datos de los patches (barras)
    bars_data = []
    all_heights = []
    all_positions = []
    categories = []
    
    for i, patch in enumerate(ax.patches):
        if hasattr(patch, "get_height") and hasattr(patch, "get_x"):
            height = float(patch.get_height())
            x_pos = float(patch.get_x())
            width = float(patch.get_width())
            
            bars_data.append({
                "index": i,
                "height": height,
                "x_position": x_pos,
                "width": width,
                "x_center": x_pos + width/2
            })
            
            all_heights.append(height)
            all_positions.append(x_pos + width/2)
    
    # Obtener etiquetas categóricas
    try:
        tick_labels = [label.get_text() for label in ax.get_xticklabels()]
        categories = [label for label in tick_labels if label.strip()]
    except:
        categories = [f"Cat_{i}" for i in range(len(bars_data))]
    
    section["bars"] = bars_data
    section["categories"] = categories
    
    # Análisis estadístico
    if all_heights:
        heights_array = np.array(all_heights)
        
        # Estadísticas básicas
        stats = {
            "mean": float(np.nanmean(heights_array)),
            "median": float(np.nanmedian(heights_array)),
            "std": float(np.nanstd(heights_array)),
            "min": float(np.nanmin(heights_array)),
            "max": float(np.nanmax(heights_array)),
            "range": float(np.nanmax(heights_array) - np.nanmin(heights_array)),
            "data_points": int(len(all_heights)),
            "total_sum": float(np.sum(heights_array))
        }
        
        # Análisis de patrones para datos categóricos
        pattern_info = {
            "pattern_type": "categorical_distribution",
            "confidence_score": 0.9,
            "equation_estimate": None,  # No aplica para datos categóricos
            "distribution_characteristics": {
                "most_frequent_category": categories[np.argmax(heights_array)] if categories and len(categories) == len(heights_array) else None,
                "least_frequent_category": categories[np.argmin(heights_array)] if categories and len(categories) == len(heights_array) else None,
                "is_uniform": bool(np.std(heights_array) < 0.1 * np.mean(heights_array)),
                "dominance_ratio": float(np.max(heights_array) / np.mean(heights_array)) if np.mean(heights_array) > 0 else 0
            }
        }
        
        # Análisis de características de forma para datos categóricos
        shape_chars = {}
        
        # 1. Monotonicity (orden de las barras)
        if len(heights_array) > 1:
            sorted_indices = np.argsort(heights_array)
            if np.array_equal(sorted_indices, np.arange(len(heights_array))):
                monotonicity = "increasing"
            elif np.array_equal(sorted_indices, np.arange(len(heights_array) - 1, -1, -1)):
                monotonicity = "decreasing"
            else:
                monotonicity = "mixed"
        else:
            monotonicity = "single_value"
        
        # 2. Smoothness (variabilidad entre barras adyacentes)
        if len(heights_array) > 2:
            adjacent_diffs = np.abs(np.diff(heights_array))
            smoothness_var = np.var(adjacent_diffs)
            if smoothness_var < 0.1 * np.var(heights_array):
                smoothness = "smooth"
            elif smoothness_var < 0.5 * np.var(heights_array):
                smoothness = "piecewise"
            else:
                smoothness = "discrete"
        else:
            smoothness = "discrete"
        
        # 3. Symmetry (distribución simétrica de valores)
        if len(heights_array) > 2:
            center_idx = len(heights_array) // 2
            if len(heights_array) % 2 == 0:
                left_half = heights_array[:center_idx]
                right_half = heights_array[center_idx:]
            else:
                left_half = heights_array[:center_idx]
                right_half = heights_array[center_idx + 1:]
            
            if len(left_half) == len(right_half):
                symmetry_corr = np.corrcoef(left_half, right_half[::-1])[0,1] if len(left_half) > 1 else 0
                symmetry = "symmetric" if abs(symmetry_corr) > 0.7 else "asymmetric"
            else:
                symmetry = "asymmetric"
        else:
            symmetry = "asymmetric"
        
        # 4. Continuity (siempre discreto para datos categóricos)
        continuity = "discrete"
        
        shape_chars = {
            "monotonicity": monotonicity,
            "smoothness": smoothness,
            "symmetry": symmetry,
            "continuity": continuity,
            "spread": float(np.max(heights_array) - np.min(heights_array)),
            "skewness": float((np.mean(heights_array) - np.median(heights_array)) / np.std(heights_array)) if np.std(heights_array) > 0 else 0
        }
        
        pattern_info["shape_characteristics"] = shape_chars
        
        section["statistics"] = stats
        section["pattern"] = pattern_info
    
    # Información del dominio
    domain_context = {
        "likely_domain": "categorical_analysis",
        "purpose": "comparison",
        "complexity_level": "medium" if len(bars_data) > 5 else "low",
        "analysis_type": "univariate_categorical"
    }
    section["domain_context"] = domain_context
    
    # Descripción para LLM
    if section.get("statistics"):
        max_category = section["pattern"]["distribution_characteristics"]["most_frequent_category"]
        description = f"Bar chart comparing {len(categories)} categories with highest value in '{max_category}'" if max_category else f"Bar chart with {len(categories)} categories"
    else:
        description = "Bar chart visualization"
    
    llm_description = {
        "one_sentence_summary": description,
        "structured_analysis": {
            "what": f"Bar chart with {len(categories)} categorical variables",
            "when": "Categorical comparison analysis",
            "why": "To compare values across different categories",
            "how": "Vertical bars representing values for each category"
        },
        "key_insights": [
            f"Highest category: {section.get('pattern', {}).get('distribution_characteristics', {}).get('most_frequent_category', 'unknown')}",
            f"Data range: {section.get('statistics', {}).get('min', 'N/A')} to {section.get('statistics', {}).get('max', 'N/A')}",
            f"Distribution pattern: {section.get('pattern', {}).get('shape_characteristics', {}).get('monotonicity', 'unknown')}"
        ] if section.get("statistics") else []
    }
    section["llm_description"] = llm_description
    
    # Contexto para LLM
    llm_context = {
        "interpretation_hints": [
            "Compare values across categories",
            "Identify the highest and lowest performing categories",
            "Look for patterns in the distribution"
        ],
        "analysis_suggestions": [
            "Rank categories by performance",
            "Calculate percentage distribution",
            "Identify outliers or unusual patterns"
        ],
        "common_questions": [
            "Which category has the highest/lowest value?",
            "What is the distribution pattern across categories?",
            "Are there significant differences between categories?"
        ],
        "related_concepts": [
            "categorical analysis",
            "comparative statistics",
            "distribution analysis",
            "frequency analysis"
        ]
    }
    section["llm_context"] = llm_context
    
    # Ranking de categorías
    if categories and all_heights:
        total = sum(all_heights)
        ranking = sorted(zip(categories, all_heights), key=lambda x: x[1], reverse=True)
        category_ranking = [
            {"category": cat, "value": val, "percentage": round(100 * val / total, 1) if total > 0 else 0, "rank": i+1}
            for i, (cat, val) in enumerate(ranking)
        ]
        # Diversidad de Shannon (entropy)
        import math
        proportions = [h/total for h in all_heights if total > 0]
        entropy = -sum(p * math.log(p) for p in proportions if p > 0)
        # Gini coefficient
        gini = 0.0
        if len(all_heights) > 1 and total > 0:
            sorted_vals = sorted(all_heights)
            n = len(sorted_vals)
            gini = (2 * sum((i+1) * val for i, val in enumerate(sorted_vals)) / (n * total)) - (n + 1) / n
        # Concentración: proporción de la categoría dominante
        concentration_ratio = max(proportions) if proportions else 0
        categorical_analysis = {
            "category_ranking": category_ranking,
            "gini_coefficient": round(gini, 3),
            "entropy": round(entropy, 3),
            "diversity_index": -sum(p * math.log(p) for p in proportions if p > 0) / math.log(len(proportions)) if len(proportions) > 1 else 0,
            "concentration_ratio": concentration_ratio
        }
        section["categorical_analysis"] = categorical_analysis
    # Fusionar categorical_analysis en statistics para que matplotlib_analyzer lo recoja
    if "statistics" in section and "categorical_analysis" in section:
        section["statistics"]["categorical_analysis"] = section["categorical_analysis"]
    # Eliminar el campo 'stats' si existe
    if "stats" in section:
        del section["stats"]
    return section 