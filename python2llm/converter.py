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
    
    def __init__(self, 
                 detail_level: str = "medium",
                 include_data: bool = True,
                 include_colors: bool = True,
                 include_statistics: bool = True):
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
        
        logger.info(f"FigureConverter initialized with detail_level={detail_level}")
    
    def convert(self, 
                figure: Any, 
                output_format: str = "text",
                **kwargs) -> Union[str, Dict, Any]:
        """
        Convert a Python figure to the specified output format.
        
        Args:
            figure: The figure object to convert (matplotlib, plotly, seaborn, etc.)
            output_format: Output format ("text", "json", "semantic")
            **kwargs: Additional arguments for specific formatters
            
        Returns:
            Converted figure in the specified format
            
        Raises:
            ValueError: If the figure type is not supported or output format is invalid
        """
        try:
            # Validate output format
            if not validate_output_format(output_format):
                raise ValueError(f"Unsupported output format: {output_format}")
            
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
                include_statistics=self.include_statistics
            )
            
            # Convert to specified format
            if output_format == "text":
                return self.text_formatter.format(analysis, **kwargs)
            elif output_format == "json":
                return self.json_formatter.format(analysis, **kwargs)
            elif output_format == "semantic":
                return self.semantic_formatter.format(analysis, **kwargs)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            logger.error(f"Error converting figure: {str(e)}")
            raise
    
    def get_supported_formats(self) -> list:
        """Get list of supported output formats."""
        return ["text", "json", "semantic"]
    
    def get_supported_libraries(self) -> list:
        """Get list of supported Python visualization libraries."""
        return ["matplotlib", "seaborn", "plotly", "bokeh", "altair", "pandas"] 