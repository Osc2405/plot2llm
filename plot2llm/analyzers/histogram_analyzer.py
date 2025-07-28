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
            "total_observations": int(total_count),
            "number_of_bins": int(len(bins_data)),
            "mean_frequency": float(np.mean(heights_array)),
            "median_frequency": float(np.median(heights_array)),
            "std_frequency": float(np.std(heights_array)),
            "min_frequency": float(np.min(heights_array)),
            "max_frequency": float(np.max(heights_array)),
            "range_frequency": float(np.max(heights_array) - np.min(heights_array))
        }
        
        # Estimación de estadísticas de los datos originales
        if total_count > 0:
            # Aproximar media y desviación estándar de los datos originales
            weighted_mean = float(np.sum(centers * heights_array) / total_count)
            weighted_variance = float(np.sum(heights_array * (centers - weighted_mean)**2) / total_count)
            weighted_std = float(np.sqrt(weighted_variance))
            
            stats["estimated_data_mean"] = weighted_mean
            stats["estimated_data_std"] = weighted_std
            stats["data_range"] = [float(min(all_edges)), float(max(all_edges))]
        
        # Análisis de patrones para histogramas
        pattern_info = {
            "pattern_type": "frequency_distribution",
            "confidence_score": 0.8,
            "equation_estimate": None,  # Podríamos estimar una distribución
            "distribution_characteristics": {}
        }
        
        # Características de la distribución
        if len(heights_array) > 2:
            # Nueva lógica mejorada para detección de distribución
            max_height = np.max(heights_array)
            
            # Función para encontrar picos locales
            def find_peaks(heights, min_height_ratio=0.25):
                """Encuentra picos locales en el histograma"""
                peaks = []
                max_height = np.max(heights)
                mean_height = np.mean(heights)
                
                # Calcular la varianza para determinar qué tan "ruidoso" es el histograma
                variance = np.var(heights)
                cv = np.sqrt(variance) / mean_height if mean_height > 0 else 0  # Coeficiente de variación
                
                # Ajustar umbrales basados en las características del histograma
                if cv > 0.8:  # Histograma muy variable (posiblemente multimodal)
                    height_threshold = 0.15  # Umbral más bajo
                    mean_multiplier = 1.2    # Multiplicador más bajo
                elif cv > 0.5:  # Histograma moderadamente variable
                    height_threshold = 0.2   # Umbral medio
                    mean_multiplier = 1.3    # Multiplicador medio
                else:  # Histograma suave (posiblemente normal)
                    height_threshold = 0.25  # Umbral más alto
                    mean_multiplier = 1.5    # Multiplicador más alto
                
                for i in range(1, len(heights) - 1):
                    # Un pico es un punto que es mayor que sus vecinos
                    if (heights[i] > heights[i-1] and 
                        heights[i] > heights[i+1] and 
                        heights[i] > height_threshold * max_height and
                        heights[i] > mean_multiplier * mean_height):
                        peaks.append(i)
                
                # Si no encontramos suficientes picos, intentar con umbrales más bajos
                if len(peaks) <= 1 and len(heights) > 10:
                    # Verificar si hay evidencia de multimodalidad
                    sorted_heights = np.sort(heights)[::-1]  # Ordenar de mayor a menor
                    if len(sorted_heights) >= 3:
                        # Si hay al menos 3 picos significativos, podría ser multimodal
                        significant_peaks = sorted_heights[:3]
                        if all(h > 0.7 * max_height for h in significant_peaks):
                            peaks = []
                            for i in range(1, len(heights) - 1):
                                if (heights[i] > heights[i-1] and 
                                    heights[i] > heights[i+1] and 
                                    heights[i] > 0.1 * max_height and
                                    heights[i] > mean_height):
                                    peaks.append(i)
                
                return peaks
            
            # Función para calcular la separación entre picos
            def calculate_peak_separation(peaks, bin_centers):
                """Calcula la separación promedio entre picos"""
                if len(peaks) < 2:
                    return 0
                
                separations = []
                for i in range(len(peaks) - 1):
                    sep = abs(bin_centers[peaks[i+1]] - bin_centers[peaks[i]])
                    separations.append(sep)
                return np.mean(separations)
            
            # Encontrar picos significativos
            peaks = find_peaks(heights_array, min_height_ratio=0.25)
            
            # Calcular separación entre picos
            peak_separation = calculate_peak_separation(peaks, centers)
            
            # Calcular el rango total de los datos
            data_range = max(centers) - min(centers)
            
            # Determinar el tipo de distribución con lógica más robusta
            if len(peaks) <= 1:
                # Unimodal - determinar si es normal, sesgada, etc.
                max_idx = np.argmax(heights_array)
                skewness_approx = (max_idx - len(heights_array)/2) / (len(heights_array)/2)
                
                if abs(skewness_approx) < 0.3:
                    pattern_type = "normal_distribution"
                elif skewness_approx > 0.3:
                    pattern_type = "right_skewed_distribution"
                else:
                    pattern_type = "left_skewed_distribution"
                    
            elif len(peaks) == 2:
                # Bimodal - lógica más robusta
                peak_indices = sorted(peaks)
                valley_height = np.min(heights_array[peak_indices[0]:peak_indices[1]+1])
                peak_heights = [heights_array[p] for p in peaks]
                avg_peak_height = np.mean(peak_heights)
                mean_height = np.mean(heights_array)
                
                # Criterios múltiples para bimodal
                criteria_met = 0
                
                # Criterio 1: Separación de picos
                if peak_separation > 0.15 * data_range:
                    criteria_met += 1
                
                # Criterio 2: Valle significativo
                if valley_height < 0.75 * avg_peak_height:
                    criteria_met += 1
                
                # Criterio 3: Picos suficientemente altos
                if all(h > 1.3 * mean_height for h in peak_heights):
                    criteria_met += 1
                
                # Criterio 4: Picos bien definidos (altura mínima)
                if all(h > 0.2 * max_height for h in peak_heights):
                    criteria_met += 1
                
                # Necesitamos al menos 3 criterios para clasificar como multimodal
                if criteria_met >= 3:
                    pattern_type = "multimodal_distribution"
                else:
                    pattern_type = "normal_distribution"
                        
            else:  # 3 o más picos
                # Multimodal - lógica más robusta para 3+ picos
                mean_height = np.mean(heights_array)
                peak_heights = [heights_array[p] for p in peaks]
                
                # Criterios múltiples para multimodal (3+ picos)
                criteria_met = 0
                
                # Criterio 1: Separación de picos (más estricto para evitar falsos positivos)
                if peak_separation > 0.12 * data_range:  # Umbral más alto para ser más conservador
                    criteria_met += 1
                
                # Criterio 2: Picos suficientemente altos (más estricto)
                if all(h > 1.2 * mean_height for h in peak_heights):  # Umbral más alto
                    criteria_met += 1
                
                # Criterio 3: Picos bien definidos (más estricto)
                if all(h > 0.15 * max_height for h in peak_heights):  # Umbral más alto
                    criteria_met += 1
                
                # Criterio 4: Valles significativos (más estricto)
                significant_valleys = 0
                for i in range(len(peaks) - 1):
                    valley_height = np.min(heights_array[peaks[i]:peaks[i+1]+1])
                    peak_heights_pair = [heights_array[peaks[i]], heights_array[peaks[i+1]]]
                    avg_peak_height_pair = np.mean(peak_heights_pair)
                    
                    if valley_height < 0.75 * avg_peak_height_pair:  # Umbral más bajo (más estricto)
                        significant_valleys += 1
                
                if significant_valleys >= len(peaks) - 1:
                    criteria_met += 1
                
                # Criterio 5: Número de picos (más estricto)
                if len(peaks) >= 4:  # Requerir al menos 4 picos para ser más conservador
                    criteria_met += 1
                
                # Para 3+ picos, necesitamos al menos 3 criterios (más estricto)
                if criteria_met >= 3:
                    pattern_type = "multimodal_distribution"
                else:
                    pattern_type = "normal_distribution"
            
            # Calcular características adicionales
            is_unimodal = len(peaks) <= 1
            skewness_approx = (np.argmax(heights_array) - len(heights_array)/2) / (len(heights_array)/2)
            
            # Calcular tendencia central aproximada
            if total_count > 0:
                central_tendency = float(np.sum(centers * heights_array) / total_count)
                spread = float(np.sqrt(np.sum(heights_array * (centers - central_tendency)**2) / total_count))
            else:
                central_tendency = float(np.mean(centers)) if len(centers) > 0 else 0
                spread = float(np.std(centers)) if len(centers) > 0 else 0
            
            pattern_info["pattern_type"] = pattern_type
            pattern_info["distribution_characteristics"] = {
                "shape": pattern_type.replace("_distribution", ""),
                "is_unimodal": is_unimodal,
                "central_tendency": central_tendency,
                "spread": spread,
                "is_symmetric": abs(skewness_approx) < 0.2,
                "tail_behavior": "heavy_tailed" if abs(skewness_approx) > 0.5 else "normal",
                "peak_bin": int(np.argmax(heights_array)),
                "peak_frequency": float(max_height),
                "skewness_estimate": float(skewness_approx),
                "outlier_bins": int(np.sum(heights_array < 0.1 * np.mean(heights_array))),
                "number_of_peaks": len(peaks),
                "peak_separation": float(peak_separation),
                "data_range": float(data_range)
            }
        
        # Análisis de características de forma
        shape_chars = {}
        
        # 1. Monotonicity (patrón de las frecuencias)
        if len(heights_array) > 1:
            diff_heights = np.diff(heights_array)
            increasing = np.sum(diff_heights > 0)
            decreasing = np.sum(diff_heights < 0)
            total_changes = len(diff_heights)
            
            if increasing > 0.7 * total_changes:
                monotonicity = "increasing"
            elif decreasing > 0.7 * total_changes:
                monotonicity = "decreasing"
            else:
                monotonicity = "mixed"
        else:
            monotonicity = "single_bin"
        
        # 2. Smoothness (suavidad de la distribución)
        if len(heights_array) > 2:
            second_diff = np.diff(heights_array, n=2)
            smoothness_var = np.var(second_diff) / np.var(heights_array) if np.var(heights_array) > 0 else 0
            
            if smoothness_var < 0.1:
                smoothness = "smooth"
            elif smoothness_var < 0.5:
                smoothness = "piecewise"
            else:
                smoothness = "discrete"
        else:
            smoothness = "discrete"
        
        # 3. Symmetry (simetría de la distribución)
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
        
        # 4. Continuity (continuidad de bins)
        if bins_data:
            bin_widths = [bin_data["bin_width"] for bin_data in bins_data]
            uniform_bins = np.std(bin_widths) < 0.01 * np.mean(bin_widths)
            continuity = "continuous" if uniform_bins else "discontinuous"
        else:
            continuity = "continuous"
        
        shape_chars = {
            "monotonicity": monotonicity,
            "smoothness": smoothness,
            "symmetry": symmetry,
            "continuity": continuity,
            "spread": float(np.max(heights_array) - np.min(heights_array)),
            "concentration": float(np.max(heights_array) / np.mean(heights_array)) if np.mean(heights_array) > 0 else 0
        }
        
        pattern_info["shape_characteristics"] = shape_chars
        
        section["statistics"] = stats
        section["pattern"] = pattern_info
    
    # Información del dominio
    domain_context = {
        "likely_domain": "statistical_analysis",
        "purpose": "distribution_visualization",
        "complexity_level": "medium" if len(bins_data) > 10 else "low",
        "analysis_type": "distribution_analysis"
    }
    section["domain_context"] = domain_context
    
    # Descripción para LLM
    if section.get("statistics"):
        pattern_type = section["pattern"]["pattern_type"]
        total_obs = section["statistics"]["total_observations"]
        central_val = section["pattern"]["distribution_characteristics"].get("central_tendency", 0)
        description = f"Histogram showing {pattern_type.replace('_', ' ')} with {total_obs} observations, centered around {central_val:.2f}"
    else:
        description = "Histogram visualization"
    
    llm_description = {
        "one_sentence_summary": description,
        "structured_analysis": {
            "what": f"Histogram with {len(bins_data)} bins showing statistical distribution",
            "when": "Distribution analysis of continuous variable",
            "why": "To visualize the shape, central tendency, and spread of a dataset",
            "how": "Frequency bars representing data density across value ranges"
        },
        "key_insights": [
            f"Distribution type: {section.get('pattern', {}).get('pattern_type', 'unknown').replace('_', ' ')}",
            f"Central tendency: {section.get('pattern', {}).get('distribution_characteristics', {}).get('central_tendency', 'N/A')}",
            f"Symmetry: {'Symmetric' if section.get('pattern', {}).get('distribution_characteristics', {}).get('is_symmetric') else 'Asymmetric'}"
        ] if section.get("statistics") else []
    }
    section["llm_description"] = llm_description
    
    # Contexto para LLM
    llm_context = {
        "interpretation_hints": [
            "Analyze the distributional shape and statistical properties",
            "Identify central tendency, spread, and skewness",
            "Look for normality, outliers, or multimodal patterns"
        ],
        "analysis_suggestions": [
            "Assess if the distribution follows a known statistical pattern",
            "Calculate descriptive statistics from the distribution",
            "Identify potential data quality issues or interesting patterns"
        ],
        "common_questions": [
            "What type of statistical distribution does this represent?",
            "Is the data normally distributed or skewed?",
            "What are the central tendency and variability of the data?"
        ],
        "related_concepts": [
            "statistical distribution",
            "probability density function",
            "descriptive statistics",
            "data distribution analysis",
            "central limit theorem"
        ]
    }
    section["llm_context"] = llm_context
    
    # Calcular parámetros básicos
    mean = float(np.mean(samples)) if 'samples' in locals() else float(stats.get('estimated_data_mean', 0))
    std = float(np.std(samples)) if 'samples' in locals() else float(stats.get('estimated_data_std', 0))
    skewness = float((np.mean((heights_array - np.mean(heights_array))**3) / (np.std(heights_array)**3)) if np.std(heights_array) > 0 else 0)
    # Goodness of fit (placeholder)
    distribution_analysis = {
        "estimated_distribution": "normal",
        "parameters": {
            "mean": mean,
            "std": std,
            "skewness": skewness
        },
        "goodness_of_fit": {
            "normal": {"p_value": 0.15, "fits_well": True},
            "uniform": {"p_value": 0.001, "fits_well": False}
        }
    }
    section["distribution_analysis"] = distribution_analysis
    
    # Intentar reconstruir muestra aproximada si no existe 'samples'
    data_for_test = None
    if 'samples' in locals():
        data_for_test = samples
    elif 'centers' in locals() and 'heights_array' in locals():
        # Reconstruir muestra aproximada a partir de bins
        data_for_test = np.repeat(centers, heights_array.astype(int))
    else:
        data_for_test = None

    normality_tests = {}
    best_fit_distribution = "normal"
    fit_parameters = {"mean": mean, "std": std}
    if data_for_test is not None and len(data_for_test) > 8:
        try:
            from scipy.stats import shapiro, anderson
            # Shapiro-Wilk
            shapiro_stat, shapiro_p = shapiro(data_for_test[:5000])  # Shapiro limita a 5000 muestras
            normality_tests["shapiro_wilk"] = {"statistic": float(shapiro_stat), "p_value": float(shapiro_p)}
            # Anderson-Darling
            ad_result = anderson(data_for_test, dist='norm')
            # Tomar el valor para el 5% (aprox)
            ad_stat = float(ad_result.statistic)
            ad_p = 0.25 if ad_stat < ad_result.critical_values[2] else 0.01  # Placeholder, scipy no da p-value exacto
            normality_tests["anderson_darling"] = {"statistic": ad_stat, "p_value": ad_p}
        except Exception:
            pass
    distribution_analysis["normality_tests"] = normality_tests
    distribution_analysis["best_fit_distribution"] = best_fit_distribution
    distribution_analysis["fit_parameters"] = fit_parameters
    section["distribution_analysis"] = distribution_analysis
    
    # Fusionar distribution_analysis en statistics para que matplotlib_analyzer lo recoja
    if "statistics" in section and "distribution_analysis" in section:
        section["statistics"]["distribution_analysis"] = section["distribution_analysis"]
    # Eliminar el campo 'stats' si existe
    if "stats" in section:
        del section["stats"]
    return section 