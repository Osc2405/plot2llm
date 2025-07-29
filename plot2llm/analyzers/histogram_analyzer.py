import numpy as np
from typing import Dict, Any, List
from scipy import stats as scipy_stats
from plot2llm.utils import generate_unified_key_insights, generate_unified_interpretation_hints

def analyze(ax, x_type=None, y_type=None) -> Dict[str, Any]:
    """
    Analiza un histograma y devuelve información semántica completa.
    """
    # Información básica del eje
    section = {
        "plot_type": "histogram",
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
    
    # Extraer datos de los patches (bins del histograma)
    bins_data = []
    all_heights = []
    all_edges = []
    curve_points = []
    
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
    
    # Agregar curve_points para histograma
    for bin_data in bins_data:
        curve_points.append({
            "x": [bin_data["bin_center"]],
            "y": [bin_data["frequency"]],
            "type": "histogram",
            "bin_index": bin_data["bin_index"],
            "bin_center": bin_data["bin_center"],
            "frequency": bin_data["frequency"]
        })
    
    section["bins"] = bins_data
    section["bin_edges"] = all_edges
    section["curve_points"] = curve_points
    
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
                "total_points": int(len(bins_data)),  # Número de bins, no observaciones
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
        
        # Detectar outliers en las frecuencias de los bins
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
        
        frequency_outliers = detect_outliers(heights_array)
        stats["outliers"] = {
            "detected": bool(frequency_outliers > 0),
            "count": frequency_outliers
        }
        
        # Estimación de estadísticas de los datos originales
        if total_count > 0:
            # Aproximar media y desviación estándar de los datos originales
            weighted_mean = float(np.sum(centers * heights_array) / total_count)
            weighted_variance = float(np.sum(heights_array * (centers - weighted_mean)**2) / total_count)
            weighted_std = float(np.sqrt(weighted_variance))
            
            stats["distribution_analysis"]["estimated_data_mean"] = weighted_mean
            stats["distribution_analysis"]["estimated_data_std"] = weighted_std
            
            # Calcular skewness y kurtosis de los datos estimados
            try:
                # Crear una muestra aproximada de los datos originales
                sample_data = []
                for i, height in enumerate(heights_array):
                    if height > 0:
                        # Añadir puntos proporcionales a la frecuencia
                        num_points = int(height)
                        bin_center = centers[i]
                        sample_data.extend([bin_center] * num_points)
                
                if len(sample_data) > 3:
                    sample_array = np.array(sample_data)
                    skewness = float(scipy_stats.skew(sample_array))
                    kurtosis = float(scipy_stats.kurtosis(sample_array))
                    
                    stats["distribution_analysis"]["skewness"] = skewness
                    stats["distribution_analysis"]["kurtosis"] = kurtosis
                    
                    # Interpretación de skewness
                    if abs(skewness) < 0.5:
                        stats["distribution_analysis"]["skewness_interpretation"] = "approximately_symmetric"
                    elif skewness > 0:
                        stats["distribution_analysis"]["skewness_interpretation"] = "right_skewed"
                    else:
                        stats["distribution_analysis"]["skewness_interpretation"] = "left_skewed"
                    
                    # Interpretación de kurtosis
                    if kurtosis < -0.5:
                        stats["distribution_analysis"]["kurtosis_interpretation"] = "platykurtic"
                    elif kurtosis > 0.5:
                        stats["distribution_analysis"]["kurtosis_interpretation"] = "leptokurtic"
                    else:
                        stats["distribution_analysis"]["kurtosis_interpretation"] = "mesokurtic"
                        
            except (ValueError, np.linalg.LinAlgError):
                # Si no se puede calcular, omitir
                pass
        
        # Análisis de bins mejorado
        if len(heights_array) > 0:
            # Estadísticas de bins
            mean_frequency = float(np.mean(heights_array))
            median_frequency = float(np.median(heights_array))
            std_frequency = float(np.std(heights_array))
            
            stats["bin_analysis"] = {
                "mean_frequency": mean_frequency,
                "median_frequency": median_frequency,
                "std_frequency": std_frequency,
                "min_frequency": float(np.min(heights_array)),
                "max_frequency": float(np.max(heights_array)),
                "range_frequency": float(np.max(heights_array) - np.min(heights_array))
            }
            
            # Análisis de picos y valles
            peaks = []
            valleys = []
            
            for i in range(1, len(heights_array) - 1):
                if heights_array[i] > heights_array[i-1] and heights_array[i] > heights_array[i+1]:
                    peaks.append({
                        "index": i,
                        "frequency": float(heights_array[i]),
                        "location": float(centers[i]),
                        "prominence": float(heights_array[i] - max(heights_array[i-1], heights_array[i+1]))
                    })
                elif heights_array[i] < heights_array[i-1] and heights_array[i] < heights_array[i+1]:
                    valleys.append({
                        "index": i,
                        "frequency": float(heights_array[i]),
                        "location": float(centers[i]),
                        "depth": float(max(heights_array[i-1], heights_array[i+1]) - heights_array[i])
                    })
            
            # Filtrar picos usando la misma lógica que find_peaks
            if len(peaks) > 1:
                # Ordenar picos por frecuencia
                peaks.sort(key=lambda x: x["frequency"], reverse=True)
                main_peak_height = peaks[0]["frequency"]
                
                # Mantener solo picos significativos (≥70% del principal)
                filtered_peaks = [peaks[0]]
                for peak in peaks[1:]:
                    if peak["frequency"] >= 0.7 * main_peak_height:
                        filtered_peaks.append(peak)
                
                peaks = filtered_peaks
            
            stats["bin_analysis"]["peaks"] = peaks
            stats["bin_analysis"]["valleys"] = valleys
            stats["bin_analysis"]["peak_count"] = len(peaks)
            stats["bin_analysis"]["valley_count"] = len(valleys)
            
            # Análisis de distribución de frecuencias
            if len(heights_array) > 1:
                # Calcular coeficiente de variación
                cv = std_frequency / mean_frequency if mean_frequency > 0 else 0
                stats["bin_analysis"]["coefficient_of_variation"] = float(cv)
                
                # Clasificar distribución de frecuencias
                if cv < 0.3:
                    stats["bin_analysis"]["frequency_distribution_type"] = "uniform"
                elif cv < 0.7:
                    stats["bin_analysis"]["frequency_distribution_type"] = "moderate_variation"
                else:
                    stats["bin_analysis"]["frequency_distribution_type"] = "high_variation"
        
        # Análisis de patrones para histogramas
        pattern_info = {
            "pattern_type": "distribution_analysis",  # Tipo base, se actualizará con el tipo específico
            "confidence_score": 0.9,
            "equation_estimate": None,  # No aplica para histogramas
            "shape_characteristics": {
                "monotonicity": "mixed",  # Histogramas pueden tener cualquier forma
                "smoothness": "discrete",  # Histogramas son discretos
                "symmetry": "unknown",  # Se determinará más adelante
                "continuity": "discontinuous"  # Histogramas son discontinuos
            },
            "distribution_characteristics": {
                "distribution_type": "unknown",  # Se determinará más adelante
                "peaks_count": 0,
                "peak_locations": [],
                "average_peak_separation": None
            }
        }
        
        # Características de la distribución
        if len(heights_array) > 2:
            # Nueva lógica mejorada para detección de distribución
            max_height = np.max(heights_array)
            
            # Función para encontrar picos locales con filtrado mejorado
            def find_peaks(heights, min_height_ratio=0.25):
                peaks = []
                for i in range(1, len(heights) - 1):
                    if heights[i] > heights[i-1] and heights[i] > heights[i+1]:
                        if heights[i] >= min_height_ratio * max_height:
                            peaks.append(i)
                
                # Si hay múltiples picos, filtrar los secundarios
                if len(peaks) > 1:
                    # Ordenar picos por altura
                    peak_heights = [heights[p] for p in peaks]
                    sorted_peaks = [p for _, p in sorted(zip(peak_heights, peaks), reverse=True)]
                    
                    # Mantener solo el pico principal y picos que sean al menos 70% de su altura
                    main_peak_height = heights[sorted_peaks[0]]
                    filtered_peaks = [sorted_peaks[0]]
                    
                    for peak in sorted_peaks[1:]:
                        if heights[peak] >= 0.7 * main_peak_height:
                            filtered_peaks.append(peak)
                    
                    return filtered_peaks
                
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
                pattern_info["pattern_type"] = "uniform_distribution"
            elif len(peaks) == 1:
                # Distribución unimodal - verificar si es normal
                peak_location = centers[peaks[0]]
                data_range = np.max(centers) - np.min(centers)
                
                # Verificar si el pico está cerca del centro (característica de distribución normal)
                center_location = (np.max(centers) + np.min(centers)) / 2
                distance_from_center = abs(peak_location - center_location) / data_range
                
                # Verificar si la distribución es simétrica alrededor del pico
                peak_idx = peaks[0]
                left_side = heights_array[:peak_idx]
                right_side = heights_array[peak_idx+1:]
                
                # Comparar lados si tienen la misma longitud
                symmetry_score = 0
                if len(left_side) > 0 and len(right_side) > 0:
                    min_len = min(len(left_side), len(right_side))
                    if min_len > 0:
                        left_compare = left_side[-min_len:]
                        right_compare = right_side[:min_len]
                        try:
                            symmetry_score = np.corrcoef(left_compare, right_compare[::-1])[0,1]
                        except:
                            symmetry_score = 0
                
                # Determinar si es normal basado en posición central y simetría
                is_normal = (distance_from_center < 0.3 and symmetry_score > 0.4)  # Más leniente
                
                if is_normal:
                    pattern_info["distribution_characteristics"]["distribution_type"] = "normal"
                    pattern_info["pattern_type"] = "normal_distribution"
                else:
                    pattern_info["distribution_characteristics"]["distribution_type"] = "unimodal"
                    pattern_info["pattern_type"] = "unimodal_distribution"
                
                pattern_info["distribution_characteristics"]["peaks_count"] = 1
                pattern_info["distribution_characteristics"]["main_peak_location"] = float(centers[peaks[0]])
            elif len(peaks) == 2:
                # Dos picos - verificar si están cerca (probablemente el mismo pico con ruido)
                peak_locations = [centers[p] for p in peaks]
                peak_separation = abs(peak_locations[1] - peak_locations[0])
                data_range = np.max(centers) - np.min(centers)
                separation_ratio = peak_separation / data_range
                
                # Si los picos están muy cerca (< 15% del rango), tratar como unimodal
                if separation_ratio < 0.15:
                    # Usar el pico más alto como principal
                    peak_heights = [heights_array[p] for p in peaks]
                    main_peak_idx = peaks[0] if peak_heights[0] > peak_heights[1] else peaks[1]
                    main_peak_location = centers[main_peak_idx]
                    
                    # Verificar si es normal
                    center_location = (np.max(centers) + np.min(centers)) / 2
                    distance_from_center = abs(main_peak_location - center_location) / data_range
                    
                    # Verificar simetría
                    left_side = heights_array[:main_peak_idx]
                    right_side = heights_array[main_peak_idx+1:]
                    symmetry_score = 0
                    if len(left_side) > 0 and len(right_side) > 0:
                        min_len = min(len(left_side), len(right_side))
                        if min_len > 0:
                            left_compare = left_side[-min_len:]
                            right_compare = right_side[:min_len]
                            try:
                                symmetry_score = np.corrcoef(left_compare, right_compare[::-1])[0,1]
                            except:
                                symmetry_score = 0
                    
                    is_normal = (distance_from_center < 0.3 and symmetry_score > 0.4)
                    
                    if is_normal:
                        pattern_info["distribution_characteristics"]["distribution_type"] = "normal"
                        pattern_info["pattern_type"] = "normal_distribution"
                    else:
                        pattern_info["distribution_characteristics"]["distribution_type"] = "unimodal"
                        pattern_info["pattern_type"] = "unimodal_distribution"
                    
                    pattern_info["distribution_characteristics"]["peaks_count"] = 1
                    pattern_info["distribution_characteristics"]["main_peak_location"] = float(main_peak_location)
                else:
                    # Picos separados - verdaderamente multimodal
                    pattern_info["distribution_characteristics"]["distribution_type"] = "multimodal"
                    pattern_info["distribution_characteristics"]["peaks_count"] = 2
                    pattern_info["distribution_characteristics"]["peak_locations"] = [float(centers[p]) for p in peaks]
                    pattern_info["distribution_characteristics"]["average_peak_separation"] = peak_separation
                    pattern_info["pattern_type"] = "multimodal_distribution"
            else:
                # Distribución multimodal - verificar si es realmente multimodal o si hay un pico dominante
                peak_heights = [heights_array[p] for p in peaks]
                max_peak_height = max(peak_heights)
                
                # Si el pico más alto es significativamente mayor que los otros, tratar como unimodal
                significant_peaks = [p for p in peaks if heights_array[p] >= 0.8 * max_peak_height]
                
                if len(significant_peaks) == 1:
                    # Solo un pico significativo - tratar como unimodal/normal
                    main_peak = significant_peaks[0]
                    peak_location = centers[main_peak]
                    data_range = np.max(centers) - np.min(centers)
                    center_location = (np.max(centers) + np.min(centers)) / 2
                    distance_from_center = abs(peak_location - center_location) / data_range
                    
                    # Verificar simetría
                    peak_idx = main_peak
                    left_side = heights_array[:peak_idx]
                    right_side = heights_array[peak_idx+1:]
                    symmetry_score = 0
                    if len(left_side) > 0 and len(right_side) > 0:
                        min_len = min(len(left_side), len(right_side))
                        if min_len > 0:
                            left_compare = left_side[-min_len:]
                            right_compare = right_side[:min_len]
                            try:
                                symmetry_score = np.corrcoef(left_compare, right_compare[::-1])[0,1]
                            except:
                                symmetry_score = 0
                    
                    is_normal = (distance_from_center < 0.3 and symmetry_score > 0.4)
                    
                    if is_normal:
                        pattern_info["distribution_characteristics"]["distribution_type"] = "normal"
                        pattern_info["pattern_type"] = "normal_distribution"
                    else:
                        pattern_info["distribution_characteristics"]["distribution_type"] = "unimodal"
                        pattern_info["pattern_type"] = "unimodal_distribution"
                    
                    pattern_info["distribution_characteristics"]["peaks_count"] = 1
                    pattern_info["distribution_characteristics"]["main_peak_location"] = float(centers[main_peak])
                else:
                    # Verdaderamente multimodal
                    pattern_info["distribution_characteristics"]["distribution_type"] = "multimodal"
                    pattern_info["distribution_characteristics"]["peaks_count"] = len(peaks)
                    pattern_info["distribution_characteristics"]["peak_locations"] = [float(centers[p]) for p in peaks]
                    pattern_info["distribution_characteristics"]["average_peak_separation"] = calculate_peak_separation(peaks, centers)
                    pattern_info["pattern_type"] = "multimodal_distribution"
            
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
                try:
                    trend_slope = np.polyfit(range(len(heights_array)), heights_array, 1)[0]
                    if trend_slope > 0.1 * np.mean(heights_array):
                        pattern_info["shape_characteristics"]["monotonicity"] = "increasing"
                    elif trend_slope < -0.1 * np.mean(heights_array):
                        pattern_info["shape_characteristics"]["monotonicity"] = "decreasing"
                    else:
                        pattern_info["shape_characteristics"]["monotonicity"] = "mixed"
                except (np.linalg.LinAlgError, ValueError):
                    pass
        
        section["stats"] = stats
        section["pattern"] = pattern_info
    
    # Generate LLM description and context
    distribution_type = pattern_info.get("distribution_characteristics", {}).get("distribution_type", "unknown") if 'pattern_info' in locals() else "unknown"
    skewness_interpretation = stats.get("distribution_analysis", {}).get("skewness_interpretation", "unknown") if 'stats' in locals() else "unknown"
    kurtosis_interpretation = stats.get("distribution_analysis", {}).get("kurtosis_interpretation", "unknown") if 'stats' in locals() else "unknown"
    total_bins = len(bins_data) if 'bins_data' in locals() else 0
    
    section["llm_description"] = {
        "one_sentence_summary": f"This histogram shows a {distribution_type} distribution with {skewness_interpretation} skewness and {kurtosis_interpretation} kurtosis.",
        "structured_analysis": {
            "what": "Histogram visualization",
            "when": "Distribution analysis",
            "why": "Frequency distribution and shape analysis",
            "how": "Through bin-based frequency representation"
        },
        "key_insights": generate_unified_key_insights({
            "distribution_type": distribution_type,
            "skewness": skewness_interpretation,
            "kurtosis": kurtosis_interpretation,
            "bin_count": total_bins
        })
    }
    
    section["llm_context"] = {
        "interpretation_hints": generate_unified_interpretation_hints({
            "shape_analysis": "Observe the distribution shape and spread.",
            "peak_analysis": "Look for peaks, valleys, and overall symmetry.",
            "statistical_analysis": "Consider the skewness and kurtosis of the distribution."
        }),
        "analysis_suggestions": [
            "Estimate skewness, kurtosis, and check for multimodality.",
            "Look for outliers in the tails of the distribution.",
            "Consider fitting a theoretical distribution to the data."
        ],
        "common_questions": [
            "Is the distribution normal, skewed, or multimodal?",
            "Are there any unusual patterns or outliers?",
            "What does the shape tell us about the underlying data?"
        ],
        "related_concepts": [
            "distribution",
            "skewness",
            "kurtosis",
            "frequency analysis"
        ]
    }
    
    return section 