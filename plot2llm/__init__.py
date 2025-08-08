"""
Plot2LLM v0.2.0 - Convert Python figures to LLM-readable formats with complete statistical analysis

This library provides tools to convert matplotlib, seaborn, plotly, and other
Python visualization figures into formats that are easily understandable by
Large Language Models (LLMs). Version 0.2.0 includes comprehensive statistical
analysis, enhanced plot type detection, and improved semantic output with
LLM-optimized descriptions and context.

Features:
- Complete statistical analysis (central tendency, variability, distribution)
- Outlier detection using IQR method
- Correlation analysis with strength and direction
- Enhanced plot type detection and classification
- Unified LLM description and context generation
- Multiple output formats (text, json, semantic)
- Support for matplotlib and seaborn visualizations
"""

__version__ = "0.2.0"
__author__ = "Plot2LLM Team"

from .analyzers import FigureAnalyzer
from .converter import FigureConverter
from .formatters import JSONFormatter, SemanticFormatter, TextFormatter

# Import sections submodule for modular semantic output

# Create a global converter instance for convenience
_converter = FigureConverter()


def convert(figure, format="text", **kwargs):
    """
    Convert a figure to the specified format with complete statistical analysis.

    This is a convenience function that uses the global FigureConverter instance.
    Version 0.2.0 includes enhanced statistical analysis and improved semantic output.

    Args:
        figure: Figure from matplotlib, seaborn, plotly, etc.
        format (str): Output format ('text', 'json', 'semantic')
        detail_level (str, optional): Analysis detail level ('low', 'medium', 'high'). Default: 'medium'
        include_statistics (bool, optional): Include statistical analysis. Default: True
        include_visual_info (bool, optional): Include visual information. Default: True
        include_data_analysis (bool, optional): Include data analysis. Default: True
        include_curve_points (bool, optional): Include raw data points for detailed analysis. Default: False
        **kwargs: Additional arguments passed to the converter

    Returns:
        str or dict: Converted data in the specified format with statistical insights

    Example:
        >>> import matplotlib.pyplot as plt
        >>> import plot2llm
        >>>
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 4, 2])
        >>>
        >>> # Basic conversion
        >>> result = plot2llm.convert(fig)
        >>>
        >>> # Conversion with enhanced statistical analysis
        >>> result = plot2llm.convert(
        ...     fig,
        ...     format='semantic',
        ...     detail_level='high',
        ...     include_statistics=True,
        ...     include_curve_points=True
        ... )
    """
    return _converter.convert(figure, output_format=format, **kwargs)


__all__ = [
    "convert",
    "FigureConverter",
    "FigureAnalyzer",
    "TextFormatter",
    "JSONFormatter",
    "SemanticFormatter",
]
