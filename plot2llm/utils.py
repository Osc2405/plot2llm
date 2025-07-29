"""
Utility functions for the plot2llm library.

This module contains helper functions for figure detection, validation,
and other common operations used throughout the library.
"""

import logging
from typing import Any, List, Union
import os
import json
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)


def detect_figure_type(figure: Any) -> str:
    """
    Detect the type of figure object.

    Args:
        figure: The figure object to analyze

    Returns:
        String indicating the figure type
    """
    try:
        # Check for seaborn figures FIRST (before matplotlib)
        if hasattr(figure, "__class__"):
            module_name = figure.__class__.__module__

            if "seaborn" in module_name:
                return "seaborn"

        # Check for matplotlib figures that contain seaborn elements
        if hasattr(figure, "_suptitle") or hasattr(figure, "axes"):
            # Check if any axis contains seaborn-specific elements
            if hasattr(figure, "axes"):
                for ax in figure.axes:
                    # Check for QuadMesh (seaborn heatmaps)
                    for collection in ax.collections:
                        if collection.__class__.__name__ == "QuadMesh":
                            return "seaborn"

                    # Check for seaborn-specific plot types
                    if hasattr(ax, "get_children"):
                        for child in ax.get_children():
                            if hasattr(child, "__class__"):
                                child_class = child.__class__.__name__
                                if child_class in [
                                    "FacetGrid",
                                    "PairGrid",
                                    "JointGrid",
                                ]:
                                    return "seaborn"

            return "matplotlib"

        # Check for seaborn figures (which are matplotlib figures)
        if hasattr(figure, "figure") and hasattr(figure.figure, "axes"):
            return "seaborn"

        # Check for plotly figures
        if hasattr(figure, "to_dict") and hasattr(figure, "data"):
            return "plotly"

        # Check for bokeh figures
        if hasattr(figure, "renderers") and hasattr(figure, "plot"):
            return "bokeh"

        # Check for altair figures
        if hasattr(figure, "to_dict") and hasattr(figure, "mark"):
            return "altair"

        # Check for pandas plotting (which returns matplotlib axes)
        if hasattr(figure, "figure") and hasattr(figure, "get_xlabel"):
            return "pandas"

        # Default to unknown
        return "unknown"

    except Exception as e:
        logger.warning(f"Error detecting figure type: {str(e)}")
        return "unknown"


def validate_output_format(output_format: str) -> bool:
    """
    Validate that the output format is supported.

    Args:
        output_format: The output format to validate

    Returns:
        True if the format is supported, False otherwise
    """
    supported_formats = ["text", "json", "semantic"]
    return output_format in supported_formats


def validate_detail_level(detail_level: str) -> bool:
    """
    Validate that the detail level is supported.

    Args:
        detail_level: The detail level to validate

    Returns:
        True if the detail level is supported, False otherwise
    """
    supported_levels = ["low", "medium", "high"]
    return detail_level in supported_levels


def serialize_axis_values(x: Union[List, Any]) -> List[str]:
    """
    Serializa valores de eje para JSON/texto. Convierte fechas a string legible.
    Args:
        x: array/list/Series de valores
    Returns:
        Lista de valores serializados
    """
    import numpy as np
    import pandas as pd

    if isinstance(x, (pd.Series, np.ndarray, list)):
        arr = np.array(x)
        if np.issubdtype(arr.dtype, np.datetime64):
            return [str(pd.to_datetime(val).date()) for val in arr]
        elif arr.dtype.kind == "O" and len(arr) > 0 and isinstance(arr[0], pd.Period):
            return [str(val) for val in arr]
        else:
            return arr.tolist()
    return list(x)


def validate_semantic_output(output: dict, schema_path: str = None) -> bool:
    """
    Valida el output semántico contra el schema JSON.
    Lanza ValidationError si no es válido.
    """
    if schema_path is None:
        schema_path = os.path.join(os.path.dirname(__file__), '../schemas/semantic_output_schema.json')
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    validate(instance=output, schema=schema)
    return True


def generate_unified_key_insights(insights_data: dict) -> list:
    """
    Genera insights unificados en formato estructurado.
    
    Args:
        insights_data: Diccionario con datos para generar insights
        
    Returns:
        Lista de insights en formato unificado
    """
    unified_insights = []
    
    # Line plot insights
    if "trend" in insights_data and insights_data["trend"] is not None:
        unified_insights.append({
            "type": "trend",
            "description": f"Data shows {insights_data['trend']} trend",
            "value": insights_data["trend"],
            "confidence": insights_data.get("trend_confidence", 0.8)
        })
    
    if "slope" in insights_data and insights_data["slope"] is not None:
        unified_insights.append({
            "type": "slope",
            "description": f"Slope: {insights_data['slope']:.3f}",
            "value": insights_data["slope"],
            "unit": "units/step"
        })
    
    # Scatter plot insights
    if "correlation_strength" in insights_data:
        unified_insights.append({
            "type": "correlation",
            "description": f"{insights_data['correlation_strength'].title()} {insights_data.get('correlation_direction', '')} correlation",
            "value": insights_data.get("correlation_value", 0.0),
            "strength": insights_data["correlation_strength"],
            "direction": insights_data.get("correlation_direction", "none")
        })
    
    if "data_points" in insights_data:
        unified_insights.append({
            "type": "statistics",
            "description": f"Total data points: {insights_data['data_points']}",
            "value": insights_data["data_points"],
            "unit": "points"
        })
    
    # Bar chart insights
    if "highest_category" in insights_data:
        unified_insights.append({
            "type": "ranking",
            "description": f"Highest category: {insights_data['highest_category']}",
            "value": insights_data["highest_category"],
            "category": "highest"
        })
    
    if "lowest_category" in insights_data:
        unified_insights.append({
            "type": "ranking",
            "description": f"Lowest category: {insights_data['lowest_category']}",
            "value": insights_data["lowest_category"],
            "category": "lowest"
        })
    
    if "total_categories" in insights_data:
        unified_insights.append({
            "type": "statistics",
            "description": f"Total categories: {insights_data['total_categories']}",
            "value": insights_data["total_categories"],
            "unit": "categories"
        })
    
    # Histogram insights
    if "distribution_type" in insights_data:
        unified_insights.append({
            "type": "distribution",
            "description": f"Distribution type: {insights_data['distribution_type']}",
            "value": insights_data["distribution_type"]
        })
    
    if "skewness" in insights_data:
        unified_insights.append({
            "type": "distribution",
            "description": f"Skewness: {insights_data['skewness']}",
            "value": insights_data["skewness"],
            "category": "skewness"
        })
    
    if "kurtosis" in insights_data:
        unified_insights.append({
            "type": "distribution",
            "description": f"Kurtosis: {insights_data['kurtosis']}",
            "value": insights_data["kurtosis"],
            "category": "kurtosis"
        })
    
    if "bin_count" in insights_data:
        unified_insights.append({
            "type": "statistics",
            "description": f"Number of bins: {insights_data['bin_count']}",
            "value": insights_data["bin_count"],
            "unit": "bins"
        })
    
    # Pattern insights
    if "pattern_confidence" in insights_data and insights_data["pattern_confidence"] is not None:
        unified_insights.append({
            "type": "pattern",
            "description": f"Pattern confidence: {insights_data['pattern_confidence']:.2f}",
            "value": insights_data["pattern_confidence"],
            "unit": "confidence"
        })
    
    # Outlier insights
    if "outliers_count" in insights_data:
        unified_insights.append({
            "type": "outliers",
            "description": f"Found {insights_data['outliers_count']} potential outliers",
            "value": insights_data["outliers_count"],
            "unit": "outliers"
        })
    
    return unified_insights


def generate_unified_interpretation_hints(hints_data: dict) -> list:
    """
    Genera hints de interpretación unificados en formato estructurado.
    
    Args:
        hints_data: Diccionario con datos para generar hints
        
    Returns:
        Lista de hints en formato unificado
    """
    unified_hints = []
    
    # Line plot hints
    if "trend_analysis" in hints_data:
        unified_hints.append({
            "type": "trend_analysis",
            "description": hints_data["trend_analysis"],
            "priority": "high",
            "category": "line_plot"
        })
    
    if "direction_analysis" in hints_data:
        unified_hints.append({
            "type": "direction_analysis",
            "description": hints_data["direction_analysis"],
            "priority": "medium",
            "category": "line_plot"
        })
    
    if "pattern_recognition" in hints_data:
        unified_hints.append({
            "type": "pattern_recognition",
            "description": hints_data["pattern_recognition"],
            "priority": "medium",
            "category": "line_plot"
        })
    
    # Scatter plot hints
    if "cluster_analysis" in hints_data:
        unified_hints.append({
            "type": "cluster_analysis",
            "description": hints_data["cluster_analysis"],
            "priority": "high",
            "category": "scatter_plot"
        })
    
    if "outlier_detection" in hints_data:
        unified_hints.append({
            "type": "outlier_detection",
            "description": hints_data["outlier_detection"],
            "priority": "medium",
            "category": "scatter_plot"
        })
    
    if "correlation_analysis" in hints_data:
        unified_hints.append({
            "type": "correlation_analysis",
            "description": hints_data["correlation_analysis"],
            "priority": "high",
            "category": "scatter_plot"
        })
    
    # Bar chart hints
    if "categorical_comparison" in hints_data:
        unified_hints.append({
            "type": "categorical_comparison",
            "description": hints_data["categorical_comparison"],
            "priority": "high",
            "category": "bar_chart"
        })
    
    if "ranking_analysis" in hints_data:
        unified_hints.append({
            "type": "ranking_analysis",
            "description": hints_data["ranking_analysis"],
            "priority": "medium",
            "category": "bar_chart"
        })
    
    if "distribution_analysis" in hints_data:
        unified_hints.append({
            "type": "distribution_analysis",
            "description": hints_data["distribution_analysis"],
            "priority": "medium",
            "category": "bar_chart"
        })
    
    # Histogram hints
    if "shape_analysis" in hints_data:
        unified_hints.append({
            "type": "shape_analysis",
            "description": hints_data["shape_analysis"],
            "priority": "high",
            "category": "histogram"
        })
    
    if "peak_analysis" in hints_data:
        unified_hints.append({
            "type": "peak_analysis",
            "description": hints_data["peak_analysis"],
            "priority": "medium",
            "category": "histogram"
        })
    
    if "statistical_analysis" in hints_data:
        unified_hints.append({
            "type": "statistical_analysis",
            "description": hints_data["statistical_analysis"],
            "priority": "medium",
            "category": "histogram"
        })
    
    # Generic hints
    if "general_analysis" in hints_data:
        unified_hints.append({
            "type": "general_analysis",
            "description": hints_data["general_analysis"],
            "priority": "low",
            "category": "generic"
        })
    
    return unified_hints
