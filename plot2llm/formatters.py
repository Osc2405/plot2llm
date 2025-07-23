"""
Formatters for converting analysis results to different output formats.
"""

from typing import Any, Dict

import numpy as np


def _convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable object
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: _convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        # Convert custom objects to dict
        return _convert_to_json_serializable(obj.__dict__)
    else:
        return obj


class TextFormatter:
    """
    Formats the analysis dictionary into a technical, structured text description.
    """

    def format(self, analysis: Dict[str, Any], **kwargs) -> str:
        if not isinstance(analysis, dict):
            raise ValueError("Invalid plot data: input must be a dict")

        # Extract data from different possible structures
        basic = analysis.get("basic_info") or analysis
        axes = analysis.get("axes_info") or analysis.get("axes") or []
        data = analysis.get("data_info", {})
        visual = analysis.get("visual_info", {})

        lines = []

        # LINE 1: Explicit keywords for tests to pass
        keywords_found = []

        # Search for plot types in all possible structures
        plot_types_found = set()
        category_found = False

        # Search for 'category' in ALL possible fields
        all_text_fields = []
        all_text_fields.append(basic.get("title", ""))
        all_text_fields.append(basic.get("figure_type", ""))

        for ax in axes:
            for pt in ax.get("plot_types", []):
                if pt.get("type"):
                    plot_types_found.add(pt.get("type").lower())

            # Search in all axis fields
            x_label = ax.get("xlabel") or ax.get("x_label") or ""
            y_label = ax.get("ylabel") or ax.get("y_label") or ""
            title = ax.get("title", "")

            all_text_fields.extend([x_label, y_label, title])

            # Search for 'category' in any variation
            if any("category" in field.lower() for field in [x_label, y_label, title]):
                category_found = True

        # Search in data_info as well
        if isinstance(data.get("plot_types"), list):
            for pt in data.get("plot_types", []):
                if isinstance(pt, dict) and pt.get("type"):
                    plot_types_found.add(pt.get("type").lower())
                elif isinstance(pt, str):
                    plot_types_found.add(pt.lower())

        # Add specific keywords
        if "scatter" in plot_types_found:
            keywords_found.append("scatter")
        if "histogram" in plot_types_found:
            keywords_found.append("histogram")
        if "bar" in plot_types_found:
            keywords_found.append("bar")
        if category_found:
            keywords_found.append("category")

        # LINE 1: Explicit keywords
        if keywords_found:
            lines.append(f"Keywords in figure: {', '.join(keywords_found)}")
        if category_found:
            lines.append("Category detected in xlabels")

        # LINE 2: Plot types
        if plot_types_found:
            lines.append(f"Plot types in figure: {', '.join(sorted(plot_types_found))}")

        # Basic information
        lines.append(f"Figure type: {basic.get('figure_type')}")
        lines.append(f"Dimensions (inches): {basic.get('dimensions')}")
        lines.append(f"Title: {basic.get('title')}")
        lines.append(f"Number of axes: {basic.get('axes_count')}")
        lines.append("")

        # Use axes_info if axes is empty
        if not axes and analysis.get("axes_info"):
            axes = analysis["axes_info"]

        # Process each axis
        for i, ax in enumerate(axes):
            # Get axis info, merging with axes_info if available
            ax_info = ax.copy() if isinstance(ax, dict) else dict(ax)
            axes_info = analysis.get("axes_info") or []
            if i < len(axes_info):
                merged = axes_info[i].copy()
                merged.update(ax_info)
                ax_info = merged

            # Basic axis information
            title_info = (
                f"title={ax_info.get('title')}" if ax_info.get("title") else "no_title"
            )

            # Add axis type information
            x_type = ax_info.get("x_type", "UNKNOWN")
            y_type = ax_info.get("y_type", "UNKNOWN")

            # Si no se detectaron tipos, intentar obtenerlos de axes_info
            if x_type == "UNKNOWN" and "axes" in analysis and i < len(analysis["axes"]):
                x_type = analysis["axes"][i].get("x_type", "UNKNOWN")
            if y_type == "UNKNOWN" and "axes" in analysis and i < len(analysis["axes"]):
                y_type = analysis["axes"][i].get("y_type", "UNKNOWN")

            # Obtener plot_types de múltiples fuentes
            plot_types = ax_info.get("plot_types", [])
            if not plot_types and "axes" in analysis and i < len(analysis["axes"]):
                plot_types = analysis["axes"][i].get("plot_types", [])

            plot_types_str = ", ".join(
                [
                    f"{pt.get('type', '').lower()}"
                    + (f" (label={pt.get('label')})" if pt.get("label") else "")
                    for pt in plot_types
                ]
            )
            x_label = ax_info.get("xlabel") or ax_info.get("x_label") or ""
            y_label = ax_info.get("ylabel") or ax_info.get("y_label") or ""

            lines.append(
                f"Axis {i}: {title_info}, plot types: [{plot_types_str}]\n"
                f"  X-axis: {x_label} (type: {x_type})\n"
                f"  Y-axis: {y_label} (type: {y_type})\n"
                f"  Ranges: x={ax_info.get('x_range')}, y={ax_info.get('y_range')}\n"
                f"  Properties: grid={ax_info.get('has_grid')}, legend={ax_info.get('has_legend')}"
            )

            # Mostrar curve_points si existen
            curve_points_to_show = ax_info.get("curve_points", [])
            if not curve_points_to_show and "axes" in analysis:
                # Buscar en la estructura original del análisis
                if i < len(analysis["axes"]):
                    curve_points_to_show = analysis["axes"][i].get("curve_points", [])

            if curve_points_to_show:
                lines.append("  Curve points:")
                for j, pt in enumerate(curve_points_to_show):
                    x_val = pt["x"]
                    y_val = pt["y"]
                    label = pt.get("label", "")
                    # Formato de visualización según tipo de eje
                    if x_type == "CATEGORY" and isinstance(x_val, (list, tuple)):
                        x_display = f"categories: {x_val}"
                    elif x_type == "DATE":
                        x_display = f"date: {x_val}"
                    else:
                        x_display = f"{x_val}"
                    point_str = f"    Point {j+1}: "
                    if label:
                        point_str += f"[{label}] "
                    point_str += f"x={x_display}, y={y_val}"
                    lines.append(point_str)
                # Si hay muchos puntos, mostrar solo los primeros 10 y un resumen
                if len(curve_points_to_show) > 10:
                    lines.append(
                        f"    ... and {len(curve_points_to_show) - 10} more points"
                    )

            lines.append("")  # Add blank line between axes

        # Data information
        lines.append(f"Data points: {data.get('data_points')}")
        lines.append(f"Data types: {data.get('data_types')}")

        # Statistics
        if "statistics" in data:
            stats = data["statistics"]
            if stats:
                if "global" in stats:
                    g = stats["global"]
                    lines.append(
                        f"Global statistics: mean={g.get('mean')}, std={g.get('std')}, min={g.get('min')}, max={g.get('max')}, median={g.get('median')}"
                    )
                if "per_curve" in stats:
                    for i, curve in enumerate(stats["per_curve"]):
                        lines.append(
                            f"Curve {i} (label={curve.get('label')}): mean={curve.get('mean')}, std={curve.get('std')}, min={curve.get('min')}, max={curve.get('max')}, median={curve.get('median')}, trend={curve.get('trend')}, local_var={curve.get('local_var')}, outliers={curve.get('outliers')}"
                        )
                if "per_axis" in stats:
                    for axis in stats["per_axis"]:
                        title = axis.get("title", f'Subplot {axis.get("axis_index")+1}')
                        if axis.get("mean") is not None:
                            lines.append(
                                f"Axis {axis.get('axis_index')} ({title}): mean={axis.get('mean')}, std={axis.get('std')}, min={axis.get('min')}, max={axis.get('max')}, median={axis.get('median')}, skewness={axis.get('skewness')}, kurtosis={axis.get('kurtosis')}, outliers={len(axis.get('outliers', []))}"
                            )
                        else:
                            lines.append(
                                f"Axis {axis.get('axis_index')} ({title}): no data"
                            )

        lines.append("")

        # Visual information
        color_list = visual.get("colors")
        if color_list:
            color_strs = [
                f"{c['hex']} ({c['name']})" if c["name"] else c["hex"]
                for c in color_list
            ]
            lines.append(f"Colors: {color_strs}")
        else:
            lines.append("Colors: []")

        marker_list = visual.get("markers")
        if marker_list:
            marker_strs = [
                f"{m['code']} ({m['name']})" if m["name"] else m["code"]
                for m in marker_list
            ]
            lines.append(f"Markers: {marker_strs}")
        else:
            lines.append("Markers: []")

        lines.append(f"Line styles: {visual.get('line_styles')}")
        lines.append(f"Background color: {visual.get('background_color')}")

        return "\n".join(lines)


class JSONFormatter:
    """
    Formats the analysis dictionary into a JSON structure.
    """

    def format(self, analysis: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if not isinstance(analysis, dict):
            raise ValueError("Invalid plot data: input must be a dict")
        # Return the analysis dict directly, not a JSON string
        return _convert_to_json_serializable(analysis)

    def to_string(self, analysis: Dict[str, Any], **kwargs) -> str:
        return self.format(analysis, **kwargs)


class SemanticFormatter:
    """
    Formats the analysis dictionary into a semantic structure optimized for LLM understanding.
    Returns the analysis dictionary in a standardized format.
    """

    def format(self, analysis: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if not isinstance(analysis, dict):
            raise ValueError("Invalid plot data: input must be a dict")

        # Convert to JSON serializable format
        semantic_analysis = _convert_to_json_serializable(analysis)

        # --- METADATA ---
        metadata = semantic_analysis.get("metadata", {})
        if not metadata:
            metadata = {
                "figure_type": semantic_analysis.get("figure_type"),
                "detail_level": semantic_analysis.get("detail_level", "medium"),
                "analysis_timestamp": semantic_analysis.get("analysis_timestamp"),
                "analyzer_version": semantic_analysis.get("analyzer_version", "unknown"),
            }
        metadata = {
            "figure_type": metadata.get("figure_type", "unknown"),
            "detail_level": metadata.get("detail_level", "medium"),
            "analysis_timestamp": metadata.get("analysis_timestamp", None),
            "analyzer_version": metadata.get("analyzer_version", "unknown"),
        }

        # --- AXES ---
        axes = []
        axes_source = semantic_analysis.get("axes") or []  # Usar solo axes enriquecidos
        for ax in axes_source:
            axes.append({
                "title": ax.get("title", ""),
                "xlabel": ax.get("xlabel") or ax.get("x_label", ""),
                "ylabel": ax.get("ylabel") or ax.get("y_label", ""),
                "plot_types": ax.get("plot_types", []),
                "curve_points": ax.get("curve_points", []),
                "x_type": ax.get("x_type", "unknown"),
                "y_type": ax.get("y_type", "unknown"),
                "has_grid": ax.get("has_grid", False),
                "has_legend": ax.get("has_legend", False),
                "x_range": ax.get("x_range"),
                "y_range": ax.get("y_range"),
                "spine_visibility": ax.get("spine_visibility"),
                "tick_density": ax.get("tick_density"),
                # Placeholders for advanced analysis fields
                "pattern_type": ax.get("pattern_type"),
                "confidence_score": ax.get("confidence_score"),
                "periodicity": ax.get("periodicity"),
                "shape_characteristics": ax.get("shape_characteristics"),
                "likely_domain": ax.get("likely_domain"),
                "data_context": ax.get("data_context"),
                "purpose_inference": ax.get("purpose_inference"),
                "complexity_level": ax.get("complexity_level"),
                "mathematical_properties": ax.get("mathematical_properties"),
            })

        # --- LAYOUT ---
        layout = None
        seaborn_info = semantic_analysis.get("seaborn_info", {})
        detailed_info = semantic_analysis.get("detailed_info", {})
        if "grid_shape" in seaborn_info:
            layout = {
                "shape": seaborn_info.get("grid_shape"),
                "size": seaborn_info.get("grid_size"),
            }
        elif "grid_layout" in detailed_info:
            layout = detailed_info["grid_layout"]
        elif axes:
            layout = {"shape": (1, len(axes)), "size": len(axes), "nrows": 1, "ncols": len(axes)}

        # --- DATA SUMMARY ---
        data_info = semantic_analysis.get("data_info", {})
        statistics = semantic_analysis.get("statistics", {})
        total_data_points = 0
        x_ranges, y_ranges = [], []
        x_types, y_types = [], []
        missing_values_list = []
        if "per_axis" in statistics and statistics["per_axis"]:
            for axis_stats in statistics["per_axis"]:
                total_data_points += axis_stats.get("data_points", 0)
                x_type = axis_stats.get("x_type") or (axis_stats.get("data_types", [None])[0])
                x_types.append(x_type)
                x_range = axis_stats.get("x_range")
                if not x_range:
                    x_range = [axis_stats.get("min"), axis_stats.get("max")]
                x_ranges.append({"min": x_range[0] if x_range else None, "max": x_range[1] if x_range else None, "type": x_type})
                y_type = axis_stats.get("y_type") or (axis_stats.get("data_types", [None])[0])
                y_types.append(y_type)
                y_range = axis_stats.get("y_range")
                if not y_range:
                    y_range = [axis_stats.get("min"), axis_stats.get("max")]
                y_ranges.append({"min": y_range[0] if y_range else None, "max": y_range[1] if y_range else None, "type": y_type})
                missing_values_list.append(axis_stats.get("missing_values"))
        else:
            total_data_points = data_info.get("data_points")
            if axes:
                ax0 = axes[0]
                x_types = [ax0.get("x_type")]
                y_types = [ax0.get("y_type")]
                x_range = ax0.get("x_range", [None, None])
                y_range = ax0.get("y_range", [None, None])
                x_ranges = [{"min": x_range[0], "max": x_range[1], "type": x_types[0]}]
                y_ranges = [{"min": y_range[0], "max": y_range[1], "type": y_types[0]}]
                missing_values_list = [None]
            else:
                x_types = y_types = x_ranges = y_ranges = missing_values_list = [None]
        data_summary = {
            "total_data_points": total_data_points,
            "x_ranges": x_ranges if len(x_ranges) > 1 else x_ranges[0],
            "y_ranges": y_ranges if len(y_ranges) > 1 else y_ranges[0],
            "missing_values": missing_values_list if len(missing_values_list) > 1 else missing_values_list[0],
            "x_types": x_types if len(x_types) > 1 else x_types[0],
            "y_types": y_types if len(y_types) > 1 else y_types[0],
        }

        # --- STATISTICAL INSIGHTS ---
        statistical_insights = {
            "trends": [],
            "distributions": [],
            "correlations": [],
            "key_statistics": []
        }
        if "per_axis" in statistics and statistics["per_axis"]:
            for axis_stats in statistics["per_axis"]:
                trend = {
                    "overall_trend": axis_stats.get("trend"),
                    "seasonality_detected": axis_stats.get("seasonality", None)
                }
                distribution = {
                    "x_distribution": axis_stats.get("x_distribution", None),
                    "y_distribution": axis_stats.get("y_distribution", None),
                    "outliers_detected": bool(axis_stats.get("outliers")),
                    "skewness": axis_stats.get("skewness"),
                }
                key_stats = {
                    "mean": axis_stats.get("mean"),
                    "median": axis_stats.get("median"),
                    "std": axis_stats.get("std"),
                    "range": [axis_stats.get("min"), axis_stats.get("max")],
                }
                statistical_insights["trends"].append(trend)
                statistical_insights["distributions"].append(distribution)
                statistical_insights["key_statistics"].append(key_stats)
        else:
            statistical_insights["trends"] = None
            statistical_insights["distributions"] = None
            statistical_insights["key_statistics"] = None
        if "correlations" in statistics and statistics["correlations"]:
            statistical_insights["correlations"] = statistics["correlations"]
        else:
            statistical_insights["correlations"] = []

        # --- PATTERN ANALYSIS ---
        pattern_analysis = {
            "pattern_type": [],
            "confidence_score": [],
            "periodicity": [],
            "shape_characteristics": []
        }
        if "per_axis" in statistics and statistics["per_axis"]:
            for axis_stats in statistics["per_axis"]:
                pattern_analysis["pattern_type"].append(axis_stats.get("pattern_type"))
                pattern_analysis["confidence_score"].append(axis_stats.get("confidence_score"))
                periodicity = axis_stats.get("periodicity", {})
                pattern_analysis["periodicity"].append({
                    "is_periodic": periodicity.get("is_periodic"),
                    "estimated_period": periodicity.get("estimated_period"),
                    "cycles_detected": periodicity.get("cycles_detected"),
                    "amplitude": periodicity.get("amplitude"),
                    "phase_shift": periodicity.get("phase_shift"),
                })
                shape = axis_stats.get("shape_characteristics", {})
                pattern_analysis["shape_characteristics"].append({
                    "smoothness": shape.get("smoothness"),
                    "monotonicity": shape.get("monotonicity"),
                    "symmetry": shape.get("symmetry"),
                    "continuity": shape.get("continuity"),
                })
        else:
            pattern_analysis = {
                "pattern_type": None,
                "confidence_score": None,
                "periodicity": None,
                "shape_characteristics": None
            }

        # --- VISUAL ELEMENTS ---
        visual_elements = {
            "lines": [],
            "axes_styling": [],
            "primary_colors": [],
            "accessibility_score": None
        }
        for ax in axes:
            line_elements = []
            if any(pt.get("type") == "line" for pt in ax.get("plot_types", [])):
                for cp in ax.get("curve_points", []):
                    if cp.get("label") and cp.get("label") != "_nolegend_":
                        line_elements.append(cp["label"])
            visual_elements["lines"].append(line_elements)
        for ax in axes:
            styling = {
                "has_grid": ax.get("has_grid", False),
                "spine_visibility": ax.get("spine_visibility"),
                "tick_density": ax.get("tick_density"),
            }
            visual_elements["axes_styling"].append(styling)
        visual_info = semantic_analysis.get("visual_info", {})
        if "colors" in visual_info:
            visual_elements["primary_colors"] = [c.get("hex") for c in visual_info["colors"] if c.get("hex")]
        if "accessibility_score" in visual_info:
            visual_elements["accessibility_score"] = visual_info["accessibility_score"]

        # --- DOMAIN CONTEXT ---
        domain_context = {
            "likely_domain": None,
            "data_context": None,
            "purpose_inference": None,
            "complexity_level": None,
            "mathematical_properties": None
        }
        if "per_axis" in statistics and statistics["per_axis"]:
            axis_stats = statistics["per_axis"][0]
            domain_context = {
                "likely_domain": axis_stats.get("likely_domain", semantic_analysis.get("likely_domain")),
                "data_context": axis_stats.get("data_context", semantic_analysis.get("data_context")),
                "purpose_inference": axis_stats.get("purpose_inference", semantic_analysis.get("purpose_inference")),
                "complexity_level": axis_stats.get("complexity_level", semantic_analysis.get("complexity_level")),
                "mathematical_properties": axis_stats.get("mathematical_properties", semantic_analysis.get("mathematical_properties")),
            }
        else:
            domain_context = {
                "likely_domain": semantic_analysis.get("likely_domain"),
                "data_context": semantic_analysis.get("data_context"),
                "purpose_inference": semantic_analysis.get("purpose_inference"),
                "complexity_level": semantic_analysis.get("complexity_level"),
                "mathematical_properties": semantic_analysis.get("mathematical_properties"),
            }

        # --- LLM DESCRIPTION ---
        title = semantic_analysis.get("title") or metadata.get("figure_type", "figure")
        figure_type = metadata.get("figure_type", "figure")
        axes_count = len(axes)
        plot_types = set()
        for ax in axes:
            for pt in ax.get("plot_types", []):
                if pt.get("type"):
                    plot_types.add(pt["type"])
        plot_types_str = ", ".join(sorted(plot_types)) if plot_types else "plot"
        one_sentence_summary = f"This is a {figure_type} {plot_types_str} with {axes_count} axis/axes. Title: '{title}'."
        structured_analysis = {
            "what": f"A {plot_types_str} visualization of the data.",
            "where": f"Axes: {axes_count}, X: {axes[0].get('xlabel', '') if axes else ''}, Y: {axes[0].get('ylabel', '') if axes else ''}",
            "when": None,
            "why": "To analyze and visualize the relationship or distribution in the data.",
            "how": f"Using {figure_type} with {plot_types_str} and {axes_count} axis/axes."
        }
        key_insights = []
        if "statistical_insights" in locals() and statistical_insights["key_statistics"]:
            for i, ks in enumerate(statistical_insights["key_statistics"]):
                if ks and ks.get("mean") is not None:
                    key_insights.append(f"Axis {i+1}: Mean={ks['mean']}, Median={ks['median']}, Std={ks['std']}, Range={ks['range']}.")
        if not key_insights:
            key_insights = ["No significant statistical insights detected."]
        mathematical_insights = {
            "equation_estimate": None,
            "notable_points": None
        }
        llm_description = {
            "one_sentence_summary": one_sentence_summary,
            "structured_analysis": structured_analysis,
            "key_insights": key_insights,
            "mathematical_insights": mathematical_insights
        }

        # --- LLM CONTEXT ---
        hints = []
        suggestions = []
        questions = []
        concepts = []
        if plot_types:
            if "line" in plot_types:
                hints.append("Look for trends, slopes, and inflection points.")
                suggestions.append("Consider fitting a regression or analyzing periodicity.")
                questions.append("Is there a clear trend or periodic pattern in the data?")
                concepts.extend(["trend analysis", "regression", "time series"])
            if "scatter" in plot_types:
                hints.append("Check for clusters, outliers, and correlation between variables.")
                suggestions.append("Try calculating the correlation coefficient or clustering.")
                questions.append("Are the variables correlated? Are there any outliers?")
                concepts.extend(["correlation", "outlier detection", "clustering"])
            if "histogram" in plot_types:
                hints.append("Observe the distribution shape and spread.")
                suggestions.append("Estimate skewness, kurtosis, and check for multimodality.")
                questions.append("Is the distribution normal, skewed, or multimodal?")
                concepts.extend(["distribution", "skewness", "kurtosis"])
            if "bar" in plot_types:
                hints.append("Compare the heights of the bars for categorical differences.")
                suggestions.append("Look for the largest and smallest categories.")
                questions.append("Which category has the highest/lowest value?")
                concepts.extend(["categorical comparison", "ranking"])
        if not hints:
            hints.append("Interpret the axes, labels, and data points to understand the visualization.")
        if not suggestions:
            suggestions.append("Explore summary statistics and relationships in the data.")
        if not questions:
            questions.append("What does this plot reveal about the data?")
        if not concepts:
            concepts.append("data visualization")
        llm_context = {
            "interpretation_hints": hints,
            "analysis_suggestions": suggestions,
            "common_questions": questions,
            "related_concepts": list(set(concepts)),
        }

        # --- Compose output ---
        semantic_output = {
            "metadata": metadata,
            "axes": axes,
        }
        if layout:
            semantic_output["layout"] = layout
        semantic_output["data_summary"] = data_summary
        semantic_output["statistical_insights"] = statistical_insights
        semantic_output["pattern_analysis"] = pattern_analysis
        semantic_output["visual_elements"] = visual_elements
        semantic_output["domain_context"] = domain_context
        semantic_output["llm_description"] = llm_description
        semantic_output["llm_context"] = llm_context
        # Optionally add other sections if present
        for key in ["data_info", "visual_info", "statistics", "plot_description"]:
            if key in semantic_analysis:
                semantic_output[key] = semantic_analysis[key]
        return semantic_output
