"""
Main converter class for transforming Python figures to LLM-readable formats.
"""

import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import pandas as pd
import numpy as np

from .analyzers import FigureAnalyzer
from .formatters import TextFormatter, JSONFormatter, SemanticFormatter
from .utils import detect_figure_type, validate_output_format

logger = logging.getLogger(__name__)


class FigureConverter:
    """
    Main class for converting Python figures to LLM-readable formats.

    This class provides a unified interface to convert figures from various
    Python visualization libraries (matplotlib, seaborn, plotly, etc.) into
    formats that Large Language Models can understand and process.
    """

    def __init__(
        self,
        detail_level: str = "medium",
        include_data: bool = True,
        include_colors: bool = True,
        include_statistics: bool = True,
    ):
        """
        Initialize the FigureConverter.

        Args:
            detail_level: Level of detail in the output ("low", "medium", "high")
            include_data: Whether to include data statistics in the output
            include_colors: Whether to include color information
            include_statistics: Whether to include statistical information
        """
        self.detail_level = detail_level
        self.include_data = include_data
        self.include_colors = include_colors
        self.include_statistics = include_statistics

        # Initialize components
        self.analyzer = FigureAnalyzer()
        self.text_formatter = TextFormatter()
        self.json_formatter = JSONFormatter()
        self.semantic_formatter = SemanticFormatter()
        self.analyzers = {"default": self.analyzer}
        self.formatters = {
            "text": self.text_formatter,
            "json": self.json_formatter,
            "semantic": self.semantic_formatter,
        }

        logger.info(f"FigureConverter initialized with detail_level={detail_level}")

    def register_analyzer(self, name, analyzer):
        self.analyzers[name] = analyzer

    def register_formatter(self, name, formatter):
        self.formatters[name] = formatter

    def convert(
        self, figure: Any, output_format: Union[str, Any] = "text", **kwargs
    ) -> str:
        """
        Convert a Python figure to the specified output format.

        Args:
            figure: The figure object to convert (matplotlib, plotly, seaborn, etc.)
            output_format: Output format ("text", "json", "semantic") or formatter object
            **kwargs: Additional arguments for specific formatters

        Returns:
            Converted figure in the specified format

        Raises:
            ValueError: If the figure type is not supported or output format is invalid
        """
        try:
            # Handle formatter objects - check if it's a real formatter, not just a string with format method
            if hasattr(output_format, "format") and not isinstance(output_format, str):
                # It's a formatter object
                formatter = output_format
                format_name = formatter.__class__.__name__.lower().replace(
                    "formatter", ""
                )
            else:
                # It's a string format
                format_name = output_format

                # Check if it's a registered formatter first
                if format_name in self.formatters:
                    formatter = self.formatters[format_name]
                elif format_name == "text":
                    formatter = self.text_formatter
                elif format_name == "json":
                    formatter = self.json_formatter
                elif format_name == "semantic":
                    formatter = self.semantic_formatter
                else:
                    # If not found in registered formatters or default ones, validate with utils
                    if not validate_output_format(format_name):
                        raise ValueError(f"Unsupported output format: {format_name}")
                    raise ValueError(f"Unsupported output format: {format_name}")

            # Detect figure type
            figure_type = detect_figure_type(figure)
            logger.info(f"Detected figure type: {figure_type}")

            # Analyze the figure
            analysis = self.analyzer.analyze(
                figure,
                figure_type,
                detail_level=self.detail_level,
                include_data=self.include_data,
                include_colors=self.include_colors,
                include_statistics=self.include_statistics,
            )

            # Convert using the formatter
            return formatter.format(analysis, **kwargs)

        except Exception as e:
            import traceback

            logger.error(f"Error converting figure: {str(e)}\n{traceback.format_exc()}")
            raise

    def get_supported_formats(self) -> list:
        """Get list of supported output formats."""
        return ["text", "json", "semantic"]

    def get_supported_libraries(self) -> list:
        """Get list of supported Python visualization libraries."""
        return ["matplotlib", "seaborn", "plotly", "bokeh", "altair", "pandas"]


def convert(figure, format="text", **kwargs):
    """
    Global utility function to convert a figure to the desired format (text, json, semantic).
    Ensures the output includes all extracted information, including curve points.
    """
    converter = FigureConverter()
    # Detect backend if not provided
    import matplotlib.figure as mpl_figure
    import matplotlib.axes as mpl_axes
    import seaborn as sns
    import matplotlib

    backend = None
    if isinstance(figure, mpl_figure.Figure) or isinstance(figure, mpl_axes.Axes):
        # Try to detect if it's a seaborn plot (by presence of seaborn attributes)
        if isinstance(figure, mpl_figure.Figure):
            axes_to_check = getattr(figure, "axes", [])
            if axes_to_check and any(
                "seaborn" in str(type(ax)) for ax in axes_to_check
            ):
                backend = "seaborn"
            else:
                backend = "matplotlib"
        else:
            # For individual axes, default to matplotlib
            backend = "matplotlib"
    else:
        backend = "matplotlib"
    # Fallback
    if backend is None:
        backend = "matplotlib"
    # Analyze and format
    analysis = converter.analyzer.analyze(figure, backend)
    if format == "text":
        return converter.text_formatter.format(analysis, **kwargs)
    elif format == "json":
        return converter.json_formatter.format(analysis, **kwargs)
    elif format == "semantic":
        return converter.semantic_formatter.format(analysis, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


__all__ = ["FigureConverter", "convert"]
