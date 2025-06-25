"""
Matplotlib-specific analyzer for extracting information from matplotlib figures.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as mpl_figure
import matplotlib.axes as mpl_axes
from matplotlib.colors import to_hex

from .base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)


class MatplotlibAnalyzer(BaseAnalyzer):
    """
    Analyzer specifically designed for matplotlib figures.
    """
    
    def __init__(self):
        """Initialize the MatplotlibAnalyzer."""
        super().__init__()
        self.supported_types = ["matplotlib.figure.Figure", "matplotlib.axes.Axes"]
        logger.debug("MatplotlibAnalyzer initialized")
    
    def analyze(self, 
                figure: Any,
                detail_level: str = "medium",
                include_data: bool = True,
                include_colors: bool = True,
                include_statistics: bool = True) -> Dict[str, Any]:
        """
        Analyze a matplotlib figure and extract relevant information.
        
        Args:
            figure: The matplotlib figure object
            detail_level: Level of detail ("low", "medium", "high")
            include_data: Whether to include data analysis
            include_colors: Whether to include color analysis
            include_statistics: Whether to include statistical analysis
            
        Returns:
            Dictionary containing the analysis results
        """
        self.include_data = include_data
        self.include_colors = include_colors
        self.include_statistics = include_statistics
        
        try:
            # Extract basic information
            basic_info = self.extract_basic_info(figure)
            
            # Extract axes information
            axes_info = self.extract_axes_info(figure)
            
            # Extract data information
            data_info = self.extract_data_info(figure) if include_data else {}
            
            # Extract visual information
            visual_info = self.extract_visual_info(figure) if include_colors else {}
            
            # Combine all information
            analysis = {
                "basic_info": basic_info,
                "axes_info": axes_info,
                "data_info": data_info,
                "visual_info": visual_info,
            }
            
            # Add detail-specific information
            if detail_level == "high":
                analysis["detailed_info"] = self._extract_detailed_info(figure)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing matplotlib figure: {str(e)}")
            raise
    
    def _get_figure_type(self, figure: Any) -> str:
        """Get the type of the matplotlib figure."""
        if isinstance(figure, mpl_figure.Figure):
            return "matplotlib.figure"
        elif isinstance(figure, mpl_axes.Axes):
            return "matplotlib.axes"
        else:
            return "unknown"
    
    def _get_dimensions(self, figure: Any) -> Tuple[int, int]:
        """Get the dimensions of the matplotlib figure."""
        try:
            if isinstance(figure, mpl_figure.Figure):
                return figure.get_size_inches()
            elif isinstance(figure, mpl_axes.Axes):
                return figure.figure.get_size_inches()
            else:
                return (0, 0)
        except Exception:
            return (0, 0)
    
    def _get_title(self, figure: Any) -> Optional[str]:
        """Get the title of the matplotlib figure."""
        try:
            if isinstance(figure, mpl_figure.Figure):
                # Get the main title if it exists
                if figure._suptitle:
                    return figure._suptitle.get_text()
                # Get title from the first axes
                axes = figure.axes
                if axes:
                    return axes[0].get_title()
            elif isinstance(figure, mpl_axes.Axes):
                return figure.get_title()
            return None
        except Exception:
            return None
    
    def _get_axes(self, figure: Any) -> List[Any]:
        """Get all axes in the matplotlib figure."""
        try:
            if isinstance(figure, mpl_figure.Figure):
                return figure.axes
            elif isinstance(figure, mpl_axes.Axes):
                return [figure]
            else:
                return []
        except Exception:
            return []
    
    def _get_axes_count(self, figure: Any) -> int:
        """Get the number of axes in the matplotlib figure."""
        return len(self._get_axes(figure))
    
    def _get_axis_type(self, ax: Any) -> str:
        """Get the type of a matplotlib axis."""
        try:
            if hasattr(ax, 'get_xscale'):
                xscale = ax.get_xscale()
                yscale = ax.get_yscale()
                if xscale == 'log' or yscale == 'log':
                    return "log"
                elif xscale == 'symlog' or yscale == 'symlog':
                    return "symlog"
                else:
                    return "linear"
            return "unknown"
        except Exception:
            return "unknown"
    
    def _get_x_label(self, ax: Any) -> Optional[str]:
        """Get the x-axis label."""
        try:
            return ax.get_xlabel()
        except Exception:
            return None
    
    def _get_y_label(self, ax: Any) -> Optional[str]:
        """Get the y-axis label."""
        try:
            return ax.get_ylabel()
        except Exception:
            return None
    
    def _get_x_range(self, ax: Any) -> Optional[Tuple[float, float]]:
        """Get the x-axis range."""
        try:
            xmin, xmax = ax.get_xlim()
            return (float(xmin), float(xmax))
        except Exception:
            return None
    
    def _get_y_range(self, ax: Any) -> Optional[Tuple[float, float]]:
        """Get the y-axis range."""
        try:
            ymin, ymax = ax.get_ylim()
            return (float(ymin), float(ymax))
        except Exception:
            return None
    
    def _has_grid(self, ax: Any) -> bool:
        """Check if the axis has a grid."""
        try:
            return ax.get_xgrid() or ax.get_ygrid()
        except Exception:
            return False
    
    def _has_legend(self, ax: Any) -> bool:
        """Check if the axis has a legend."""
        try:
            return ax.get_legend() is not None
        except Exception:
            return False
    
    def _get_data_points(self, figure: Any) -> int:
        """Get the number of data points in the figure."""
        try:
            total_points = 0
            axes = self._get_axes(figure)
            
            for ax in axes:
                for line in ax.lines:
                    if hasattr(line, '_x') and hasattr(line, '_y'):
                        total_points += len(line._x)
                
                for collection in ax.collections:
                    if hasattr(collection, '_offsets'):
                        total_points += len(collection._offsets)
            
            return total_points
        except Exception:
            return 0
    
    def _get_data_types(self, figure: Any) -> List[str]:
        """Get the types of data in the figure."""
        data_types = []
        try:
            axes = self._get_axes(figure)
            
            for ax in axes:
                if ax.lines:
                    data_types.append("line_plot")
                if ax.collections:
                    data_types.append("scatter_plot")
                if ax.patches:
                    data_types.append("bar_plot")
                if ax.images:
                    data_types.append("image")
                if ax.texts:
                    data_types.append("text")
            
            return list(set(data_types))
        except Exception:
            return []
    
    def _get_statistics(self, figure: Any) -> Dict[str, Any]:
        """Get statistical information about the data."""
        stats = {}
        try:
            axes = self._get_axes(figure)
            
            all_data = []
            for ax in axes:
                for line in ax.lines:
                    if hasattr(line, '_y') and line._y is not None:
                        all_data.extend(line._y)
            
            if all_data:
                all_data = np.array(all_data)
                stats = {
                    "mean": float(np.mean(all_data)),
                    "std": float(np.std(all_data)),
                    "min": float(np.min(all_data)),
                    "max": float(np.max(all_data)),
                    "median": float(np.median(all_data)),
                }
        
        except Exception as e:
            logger.warning(f"Error calculating statistics: {str(e)}")
        
        return stats
    
    def _get_colors(self, figure: Any) -> List[str]:
        """Get the colors used in the figure."""
        colors = []
        try:
            axes = self._get_axes(figure)
            
            for ax in axes:
                for line in ax.lines:
                    if hasattr(line, '_color'):
                        color = to_hex(line._color)
                        if color not in colors:
                            colors.append(color)
                
                for collection in ax.collections:
                    if hasattr(collection, '_facecolors'):
                        for color in collection._facecolors:
                            hex_color = to_hex(color)
                            if hex_color not in colors:
                                colors.append(hex_color)
        
        except Exception as e:
            logger.warning(f"Error extracting colors: {str(e)}")
        
        return colors
    
    def _get_markers(self, figure: Any) -> List[str]:
        """Get the markers used in the figure."""
        markers = []
        try:
            axes = self._get_axes(figure)
            
            for ax in axes:
                for line in ax.lines:
                    if hasattr(line, '_marker') and line._marker != 'None':
                        if line._marker not in markers:
                            markers.append(line._marker)
        
        except Exception as e:
            logger.warning(f"Error extracting markers: {str(e)}")
        
        return markers
    
    def _get_line_styles(self, figure: Any) -> List[str]:
        """Get the line styles used in the figure."""
        styles = []
        try:
            axes = self._get_axes(figure)
            
            for ax in axes:
                for line in ax.lines:
                    if hasattr(line, '_linestyle') and line._linestyle != 'None':
                        if line._linestyle not in styles:
                            styles.append(line._linestyle)
        
        except Exception as e:
            logger.warning(f"Error extracting line styles: {str(e)}")
        
        return styles
    
    def _get_background_color(self, figure: Any) -> Optional[str]:
        """Get the background color of the figure."""
        try:
            if isinstance(figure, mpl_figure.Figure):
                return to_hex(figure.get_facecolor())
            elif isinstance(figure, mpl_axes.Axes):
                return to_hex(figure.get_facecolor())
            return None
        except Exception:
            return None
    
    def _extract_detailed_info(self, figure: Any) -> Dict[str, Any]:
        """Extract detailed information for high detail level."""
        detailed_info = {}
        try:
            axes = self._get_axes(figure)
            
            detailed_info["line_details"] = []
            detailed_info["collection_details"] = []
            
            for ax in axes:
                for line in ax.lines:
                    line_detail = {
                        "label": line.get_label(),
                        "color": to_hex(line._color) if hasattr(line, '_color') else None,
                        "linewidth": line.get_linewidth(),
                        "linestyle": line.get_linestyle(),
                        "marker": line.get_marker(),
                        "markersize": line.get_markersize(),
                    }
                    detailed_info["line_details"].append(line_detail)
                
                for collection in ax.collections:
                    collection_detail = {
                        "type": type(collection).__name__,
                        "alpha": collection.get_alpha(),
                        "edgecolors": [to_hex(c) for c in collection.get_edgecolors()] if hasattr(collection, 'get_edgecolors') else [],
                    }
                    detailed_info["collection_details"].append(collection_detail)
        
        except Exception as e:
            logger.warning(f"Error extracting detailed info: {str(e)}")
        
        return detailed_info