import numpy as np
from typing import Dict, Any, List

def analyze(ax, x_type=None, y_type=None) -> Dict[str, Any]:
    """
    Analiza un histograma y devuelve información semántica completa.
    """
    # Información básica del eje
    section = {
        "plot_type": "histogram",
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
    
    # Extraer datos de los patches (bins del histograma)
    bins_data = []
    all_heights = []
    all_edges = []
    
    for i, patch in enumerate(ax.patches):
        if hasattr(patch, "get_height") and hasattr(patch, "get_x"):
            height = float(patch.get_height())
            x_pos = float(patch.get_x())
            width = float(patch.get_width())
            
            bins_data.append({
                "bin_index": i,
                "frequency": height,
                "left_edge": x_pos,
                "right_edge": x_pos + width,
                "bin_center": x_pos + width/2,
                "bin_width": width
            })
            
            all_heights.append(height)
            all_edges.append(x_pos)
    
    if all_edges:
        all_edges.append(all_edges[-1] + bins_data[-1]["bin_width"])  # Añadir el último borde
    
    section["bins"] = bins_data
    section["bin_edges"] = all_edges
    
    # Análisis estadístico
    if all_heights and bins_data:
        heights_array = np.array(all_heights)
        centers = np.array([bin_data["bin_center"] for bin_data in bins_data])
        
        # Estadísticas básicas de la distribución
        total_count = np.sum(heights_array)
        
        # Estadísticas del histograma (frecuencias)
        stats = {
            "central_tendency": {
                "mean": float(np.mean(heights_array)),
                "median": float(np.median(heights_array)),
                "mode": float(heights_array[np.argmax(heights_array)]) if len(heights_array) > 0 else None
            },
            "variability": {
                "std": float(np.std(heights_array)),
                "variance": float(np.std(heights_array) ** 2),
                "range": {
                    "min": float(np.min(heights_array)),
                    "max": float(np.max(heights_array))
                }
            },
            "data_quality": {
                "total_points": int(total_count),
                "missing_values": 0  # Histogramas no tienen missing values
            },
            "distribution_analysis": {
                "total_observations": int(total_count),
                "number_of_bins": int(len(bins_data)),
                "estimated_data_mean": None,
                "estimated_data_std": None,
                "data_range": [float(min(all_edges)), float(max(all_edges))] if all_edges else None
            }
        }
        
        # Estimación de estadísticas de los datos originales
        if total_count > 0:
            # Aproximar media y desviación estándar de los datos originales
            weighted_mean = float(np.sum(centers * heights_array) / total_count)
            weighted_variance = float(np.sum(heights_array * (centers - weighted_mean)**2) / total_count)
            weighted_std = float(np.sqrt(weighted_variance))
            
            stats["distribution_analysis"]["estimated_data_mean"] = weighted_mean
            stats["distribution_analysis"]["estimated_data_std"] = weighted_std
        
        # Análisis de patrones para histogramas
        pattern_info = {
            "pattern_type": "frequency_distribution",
            "confidence_score": 0.8,
            "equation_estimate": None,  # Podríamos estimar una distribución
            "shape_characteristics": {
                "monotonicity": "mixed",  # Histogramas pueden tener cualquier forma
                "smoothness": "discrete",  # Histogramas son discretos
                "symmetry": "asymmetric",  # Histogramas raramente son simétricos
                "continuity": "discontinuous"  # Histogramas son discontinuos
            },
            "distribution_characteristics": {}
        }
        
        # Características de la distribución
        if len(heights_array) > 2:
            # Nueva lógica mejorada para detección de distribución
            max_height = np.max(heights_array)
            
            # Función para encontrar picos locales
            def find_peaks(heights, min_height_ratio=0.25):
                peaks = []
                for i in range(1, len(heights) - 1):
                    if heights[i] > heights[i-1] and heights[i] > heights[i+1]:
                        if heights[i] >= min_height_ratio * max_height:
                            peaks.append(i)
                return peaks
            
            # Función para calcular la separación entre picos
            def calculate_peak_separation(peaks, bin_centers):
                if len(peaks) < 2:
                    return 0
                separations = []
                for i in range(len(peaks) - 1):
                    sep = bin_centers[peaks[i+1]] - bin_centers[peaks[i]]
                    separations.append(sep)
                return np.mean(separations) if separations else 0
            
            # Detectar picos
            peaks = find_peaks(heights_array)
            
            # Análisis de distribución
            if len(peaks) == 0:
                # Distribución uniforme o sin picos claros
                pattern_info["distribution_characteristics"]["distribution_type"] = "uniform"
                pattern_info["distribution_characteristics"]["peaks_count"] = 0
            elif len(peaks) == 1:
                # Distribución unimodal
                pattern_info["distribution_characteristics"]["distribution_type"] = "unimodal"
                pattern_info["distribution_characteristics"]["peaks_count"] = 1
                pattern_info["distribution_characteristics"]["main_peak_location"] = float(centers[peaks[0]])
            else:
                # Distribución multimodal
                pattern_info["distribution_characteristics"]["distribution_type"] = "multimodal"
                pattern_info["distribution_characteristics"]["peaks_count"] = len(peaks)
                pattern_info["distribution_characteristics"]["peak_locations"] = [float(centers[p]) for p in peaks]
                pattern_info["distribution_characteristics"]["average_peak_separation"] = calculate_peak_separation(peaks, centers)
            
            # Análisis de simetría
            if len(heights_array) > 4:
                center_idx = len(heights_array) // 2
                left_half = heights_array[:center_idx]
                right_half = heights_array[center_idx:][:len(left_half)]
                
                if len(left_half) == len(right_half):
                    try:
                        symmetry_corr = np.corrcoef(left_half, right_half[::-1])[0,1] if len(left_half) > 1 else 0
                        pattern_info["shape_characteristics"]["symmetry"] = "symmetric" if abs(symmetry_corr) > 0.7 else "asymmetric"
                    except (np.linalg.LinAlgError, ValueError):
                        pattern_info["shape_characteristics"]["symmetry"] = "asymmetric"
                else:
                    pattern_info["shape_characteristics"]["symmetry"] = "asymmetric"
            
            # Análisis de monotonicity (tendencia general)
            if len(heights_array) > 2:
                # Calcular tendencia general del histograma
                trend_slope = np.polyfit(range(len(heights_array)), heights_array, 1)[0]
                if trend_slope > 0.1 * np.mean(heights_array):
                    pattern_info["shape_characteristics"]["monotonicity"] = "increasing"
                elif trend_slope < -0.1 * np.mean(heights_array):
                    pattern_info["shape_characteristics"]["monotonicity"] = "decreasing"
                else:
                    pattern_info["shape_characteristics"]["monotonicity"] = "mixed"
        
        section["stats"] = stats
        section["pattern"] = pattern_info
    
    return section 