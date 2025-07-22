"""
Plot2LLM - Convert Python figures to LLM-readable formats

This library provides tools to convert matplotlib, seaborn, plotly, and other
Python visualization figures into formats that are easily understandable by
Large Language Models (LLMs).
"""

__version__ = "0.1.22"
__author__ = "Plot2LLM Team"

from .analyzers import FigureAnalyzer
from .converter import FigureConverter
from .formatters import JSONFormatter, SemanticFormatter, TextFormatter

# Create a global converter instance for convenience
_converter = FigureConverter()


def convert(figure, format="text", **kwargs):
    """
    Convert a figure to the specified format.

    This is a convenience function that uses the global FigureConverter instance.

    Args:
        figure: Figure from matplotlib, seaborn, plotly, etc.
        format (str): Output format ('text', 'json', 'semantic')
        **kwargs: Additional arguments passed to the converter

    Returns:
        str or dict: Converted data in the specified format
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
