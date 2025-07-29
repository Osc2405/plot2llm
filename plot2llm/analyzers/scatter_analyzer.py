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
            "central_tendency": {
                "mean": float(np.nanmean(y_array)),
                "median": float(np.nanmedian(y_array)),
                "mode": None  # No calculamos mode para scatter
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
                "total_points": int(len(all_x_data)),
                "missing_values": int(np.sum(np.isnan(x_array) | np.isnan(y_array)))
            },
            "x_axis": {
                "mean": float(np.nanmean(x_array)),
                "std": float(np.nanstd(x_array)),
                "min": float(np.nanmin(x_array)),
                "max": float(np.nanmax(x_array))
            }
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
            "pattern_type": "correlation_analysis",
            "confidence_score": 0.9,
            "equation_estimate": None,  # No aplica para scatter
            "shape_characteristics": {
                "monotonicity": "increasing" if correlation > 0.3 else "decreasing" if correlation < -0.3 else "mixed",
                "smoothness": "discrete",  # Scatter plots son discretos
                "symmetry": "symmetric" if abs(correlation) < 0.1 else "asymmetric",
                "continuity": "discontinuous"  # Scatter plots son discontinuos
            },
            "correlation": correlation,
            "correlation_strength": (
                "strong" if abs(correlation) > 0.7 else
                "moderate" if abs(correlation) > 0.3 else
                "weak"
            ),
            "correlation_direction": (
                "positive" if correlation > 0 else
                "negative" if correlation < 0 else
                "none"
            )
        }
        
        section["stats"] = stats
        section["pattern"] = pattern_info
    
    return section 