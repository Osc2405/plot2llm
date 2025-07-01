"""
Formatters for converting analysis results to different output formats.
"""

import json
import numpy as np
from typing import Any, Dict

from .analyzers import MatplotlibAnalyzer


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
    elif hasattr(obj, '__dict__'):
        # Convert custom objects to dict
        return _convert_to_json_serializable(obj.__dict__)
    else:
        return obj


class TextFormatter:
    """
    Formats the analysis dictionary into a technical, structured text description.
    """
    def format(self, analysis: Dict[str, Any], **kwargs) -> str:
        basic = analysis.get("basic_info", {})
        axes = analysis.get("axes_info", [])
        data = analysis.get("data_info", {})
        visual = analysis.get("visual_info", {})
        lines = []
        lines.append(f"Figure type: {basic.get('figure_type')}")
        lines.append(f"Dimensions (inches): {basic.get('dimensions')}")
        lines.append(f"Title: {basic.get('title')}")
        lines.append(f"Number of axes: {basic.get('axes_count')}")
        lines.append("")
        for i, ax in enumerate(axes):
            title_info = f"title={ax.get('title')}" if ax.get('title') else "no_title"
            lines.append(f"Axis {i}: {title_info}, type={ax.get('type')}, x_label={ax.get('x_label')}, y_label={ax.get('y_label')}, x_range={ax.get('x_range')}, y_range={ax.get('y_range')}, grid={ax.get('has_grid')}, legend={ax.get('has_legend')}")
        lines.append("")
        lines.append(f"Data points: {data.get('data_points')}")
        lines.append(f"Data types: {data.get('data_types')}")
        if 'statistics' in data:
            stats = data['statistics']
            if stats:
                if 'global' in stats:
                    g = stats['global']
                    lines.append(f"Global statistics: mean={g.get('mean')}, std={g.get('std')}, min={g.get('min')}, max={g.get('max')}, median={g.get('median')}")
                if 'per_curve' in stats:
                    for i, curve in enumerate(stats['per_curve']):
                        lines.append(f"Curve {i} (label={curve.get('label')}): mean={curve.get('mean')}, std={curve.get('std')}, min={curve.get('min')}, max={curve.get('max')}, median={curve.get('median')}, trend={curve.get('trend')}, local_var={curve.get('local_var')}, outliers={curve.get('outliers')}")
                if 'per_axis' in stats:
                    for axis in stats['per_axis']:
                        title = axis.get('title', f'Subplot {axis.get("axis_index")+1}')
                        if axis.get('mean') is not None:
                            lines.append(f"Axis {axis.get('axis_index')} ({title}): mean={axis.get('mean')}, std={axis.get('std')}, min={axis.get('min')}, max={axis.get('max')}, median={axis.get('median')}, skewness={axis.get('skewness')}, kurtosis={axis.get('kurtosis')}, outliers={len(axis.get('outliers', []))}")
                        else:
                            lines.append(f"Axis {axis.get('axis_index')} ({title}): no data")
        lines.append("")
        # Colors
        color_list = visual.get('colors')
        if color_list:
            color_strs = [f"{c['hex']} ({c['name']})" if c['name'] else c['hex'] for c in color_list]
            lines.append(f"Colors: {color_strs}")
        else:
            lines.append("Colors: []")
        # Markers
        marker_list = visual.get('markers')
        if marker_list:
            marker_strs = [f"{m['code']} ({m['name']})" if m['name'] else m['code'] for m in marker_list]
            lines.append(f"Markers: {marker_strs}")
        else:
            lines.append("Markers: []")
        lines.append(f"Line styles: {visual.get('line_styles')}")
        lines.append(f"Background color: {visual.get('background_color')}")
        return '\n'.join(lines)


class JSONFormatter:
    """
    Formats the analysis dictionary into a JSON structure.
    """
    def format(self, analysis: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Format the analysis into a JSON-compatible dictionary.
        
        Args:
            analysis: The analysis dictionary to format
            **kwargs: Additional formatting options
            
        Returns:
            The analysis dictionary (JSON-compatible)
        """
        # Convert numpy arrays and other non-serializable objects
        return _convert_to_json_serializable(analysis)
    
    def to_string(self, analysis: Dict[str, Any], **kwargs) -> str:
        """
        Convert the analysis to a JSON string.
        
        Args:
            analysis: The analysis dictionary to format
            **kwargs: Additional formatting options
            
        Returns:
            JSON string representation
        """
        # Convert to JSON-serializable format first
        json_compatible = _convert_to_json_serializable(analysis)
        return json.dumps(json_compatible, indent=2, ensure_ascii=False)


class SemanticFormatter:
    """
    Formats the analysis dictionary into a semantic structure (for now, just returns the dict).
    """
    def format(self, analysis: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return analysis 