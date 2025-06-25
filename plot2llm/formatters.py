"""
Formatters for converting analysis results to different output formats.
"""

import json
from typing import Any, Dict

from .analyzers import MatplotlibAnalyzer


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
            lines.append(f"Axis {i}: type={ax.get('type')}, x_label={ax.get('x_label')}, y_label={ax.get('y_label')}, x_range={ax.get('x_range')}, y_range={ax.get('y_range')}, grid={ax.get('has_grid')}, legend={ax.get('has_legend')}")
        lines.append("")
        lines.append(f"Data points: {data.get('data_points')}")
        lines.append(f"Data types: {data.get('data_types')}")
        if 'statistics' in data:
            stats = data['statistics']
            if stats:
                lines.append(f"Statistics: mean={stats.get('mean')}, std={stats.get('std')}, min={stats.get('min')}, max={stats.get('max')}, median={stats.get('median')}")
        lines.append("")
        lines.append(f"Colors: {visual.get('colors')}")
        lines.append(f"Markers: {visual.get('markers')}")
        lines.append(f"Line styles: {visual.get('line_styles')}")
        lines.append(f"Background color: {visual.get('background_color')}")
        return '\n'.join(lines)


class JSONFormatter:
    """
    Formats the analysis dictionary into a JSON string.
    """
    def format(self, analysis: Dict[str, Any], **kwargs) -> str:
        return json.dumps(analysis, indent=2, default=str)


class SemanticFormatter:
    """
    Formats the analysis dictionary into a semantic structure (for now, just returns the dict).
    """
    def format(self, analysis: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return analysis 