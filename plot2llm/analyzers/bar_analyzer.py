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
            "central_tendency": {
                "mean": float(np.nanmean(heights_array)),
                "median": float(np.nanmedian(heights_array)),
                "mode": float(heights_array[np.argmax(heights_array)]) if len(heights_array) > 0 else None
            },
            "variability": {
                "std": float(np.nanstd(heights_array)),
                "variance": float(np.nanstd(heights_array) ** 2),
                "range": {
                    "min": float(np.nanmin(heights_array)),
                    "max": float(np.nanmax(heights_array))
                }
            },
            "data_quality": {
                "total_points": int(len(all_heights)),
                "missing_values": 0  # Barras no tienen missing values
            },
            "categorical_analysis": {
                "total_sum": float(np.sum(heights_array)),
                "most_frequent_category": categories[np.argmax(heights_array)] if categories and len(categories) == len(heights_array) else None,
                "least_frequent_category": categories[np.argmin(heights_array)] if categories and len(categories) == len(heights_array) else None
            }
        }
        
        # Análisis de patrones para datos categóricos
        pattern_info = {
            "pattern_type": "categorical_distribution",
            "confidence_score": 0.9,
            "equation_estimate": None,  # No aplica para datos categóricos
            "shape_characteristics": {
                "monotonicity": "mixed",  # Barras pueden tener cualquier orden
                "smoothness": "discrete",  # Barras son discretas
                "symmetry": "asymmetric",  # Barras raramente son simétricas
                "continuity": "discontinuous"  # Barras son discontinuas
            },
            "distribution_characteristics": {
                "is_uniform": bool(np.std(heights_array) < 0.1 * np.mean(heights_array)),
                "dominance_ratio": float(np.max(heights_array) / np.mean(heights_array)) if np.mean(heights_array) > 0 else 0
            }
        }
        
        # Análisis de características de forma para datos categóricos
        if len(heights_array) > 1:
            # 1. Monotonicity (orden de las barras)
            sorted_indices = np.argsort(heights_array)
            if np.array_equal(sorted_indices, np.arange(len(heights_array))):
                pattern_info["shape_characteristics"]["monotonicity"] = "increasing"
            elif np.array_equal(sorted_indices, np.arange(len(heights_array) - 1, -1, -1)):
                pattern_info["shape_characteristics"]["monotonicity"] = "decreasing"
            else:
                pattern_info["shape_characteristics"]["monotonicity"] = "mixed"
        
        section["stats"] = stats
        section["pattern"] = pattern_info
    
    return section 