"""
Plot2LLM - Convert Python figures to LLM-readable formats

This library provides tools to convert matplotlib, seaborn, plotly, and other
Python visualization figures into formats that are easily understandable by
Large Language Models (LLMs).
"""

__version__ = "0.1.0"
__author__ = "Plot2LLM Team"

from .converter import FigureConverter
from .analyzers import FigureAnalyzer
from .formatters import TextFormatter, JSONFormatter, SemanticFormatter

__all__ = [
    "FigureConverter",
    "FigureAnalyzer", 
    "TextFormatter",
    "JSONFormatter",
    "SemanticFormatter"
] 