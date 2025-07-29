import numpy as np
from typing import Dict, Any, List

def analyze(ax, x_type=None, y_type=None) -> Dict[str, Any]:
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
        "has_legend": bool(ax.get_legend() is not None),
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
        
        # Filtrar NaN values antes del análisis
        valid_mask = ~(np.isnan(x_array) | np.isnan(y_array))
        x_clean = x_array[valid_mask]
        y_clean = y_array[valid_mask]
        
        # Estadísticas básicas
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
                # Datos constantes o muy similares
                trend = "stable"
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
    
    return section 