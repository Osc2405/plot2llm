"""
Utility functions for the plot2llm library.

This module contains helper functions for figure detection, validation,
and other common operations used throughout the library.
"""

import logging
from typing import Any, Union
from pathlib import Path

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
        # Check for matplotlib figures
        if hasattr(figure, '_suptitle') or hasattr(figure, 'axes'):
            return "matplotlib"
        
        # Check for plotly figures
        if hasattr(figure, 'to_dict') and hasattr(figure, 'data'):
            return "plotly"
        
        # Check for seaborn figures (which are matplotlib figures)
        if hasattr(figure, 'figure') and hasattr(figure.figure, 'axes'):
            return "seaborn"
        
        # Check for bokeh figures
        if hasattr(figure, 'renderers') and hasattr(figure, 'plot'):
            return "bokeh"
        
        # Check for altair figures
        if hasattr(figure, 'to_dict') and hasattr(figure, 'mark'):
            return "altair"
        
        # Check for pandas plotting (which returns matplotlib axes)
        if hasattr(figure, 'figure') and hasattr(figure, 'get_xlabel'):
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