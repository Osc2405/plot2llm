"""
Seaborn-specific analyzer for extracting information from seaborn figures.
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


class SeabornAnalyzer(BaseAnalyzer):
    """
    Analyzer specifically designed for seaborn figures.
    
    Seaborn is built on top of matplotlib, so this analyzer extends
    matplotlib functionality with seaborn-specific features.
    """
    
    def __init__(self):
        """Initialize the SeabornAnalyzer."""
        super().__init__()
        self.supported_types = [
            "matplotlib.figure.Figure", 
            "matplotlib.axes.Axes",
            "seaborn.axisgrid.FacetGrid",
            "seaborn.axisgrid.PairGrid",
            "seaborn.axisgrid.JointGrid"
        ]
        logger.debug("SeabornAnalyzer initialized")
    
    def analyze(self, 
                figure: Any,
                detail_level: str = "medium",
                include_data: bool = True,
                include_colors: bool = True,
                include_statistics: bool = True) -> Dict[str, Any]:
        """
        Analyze a seaborn figure and extract relevant information.
        
        Args:
            figure: The seaborn figure object
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
            
            # Extract seaborn-specific information
            seaborn_info = self._extract_seaborn_info(figure)
            
            # Combine all information
            analysis = {
                "basic_info": basic_info,
                "axes_info": axes_info,
                "data_info": data_info,
                "visual_info": visual_info,
                "seaborn_info": seaborn_info,
            }
            
            # Add detail-specific information
            if detail_level == "high":
                analysis["detailed_info"] = self._extract_detailed_info(figure)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing seaborn figure: {str(e)}")
            raise
    
    def _get_figure_type(self, figure: Any) -> str:
        """Get the type of the seaborn figure."""
        try:
            # Check for seaborn-specific types
            if hasattr(figure, '__class__'):
                class_name = figure.__class__.__name__
                module_name = figure.__class__.__module__
                
                if 'seaborn' in module_name:
                    if 'FacetGrid' in class_name:
                        return "seaborn.facetgrid"
                    elif 'PairGrid' in class_name:
                        return "seaborn.pairgrid"
                    elif 'JointGrid' in class_name:
                        return "seaborn.jointgrid"
                    elif 'Heatmap' in class_name:
                        return "seaborn.heatmap"
                    elif 'ClusterGrid' in class_name:
                        return "seaborn.clustermap"
                    else:
                        return f"seaborn.{class_name.lower()}"
                
                # Fall back to matplotlib types
                if isinstance(figure, mpl_figure.Figure):
                    return "matplotlib.figure"
                elif isinstance(figure, mpl_axes.Axes):
                    return "matplotlib.axes"
            
            return "unknown"
        except Exception:
            return "unknown"
    
    def _get_dimensions(self, figure: Any) -> Tuple[int, int]:
        """Get the dimensions of the seaborn figure."""
        try:
            # Handle seaborn grid objects
            if hasattr(figure, 'fig'):
                return figure.fig.get_size_inches()
            elif hasattr(figure, 'figure'):
                return figure.figure.get_size_inches()
            elif isinstance(figure, mpl_figure.Figure):
                return figure.get_size_inches()
            elif isinstance(figure, mpl_axes.Axes):
                return figure.figure.get_size_inches()
            else:
                return (0, 0)
        except Exception:
            return (0, 0)
    
    def _get_title(self, figure: Any) -> Optional[str]:
        """Get the title of the seaborn figure."""
        try:
            # Handle seaborn grid objects
            if hasattr(figure, 'fig'):
                fig = figure.fig
                if fig._suptitle:
                    return fig._suptitle.get_text()
                if fig.axes:
                    return fig.axes[0].get_title()
            elif hasattr(figure, 'figure'):
                fig = figure.figure
                if fig._suptitle:
                    return fig._suptitle.get_text()
                if fig.axes:
                    return fig.axes[0].get_title()
            elif isinstance(figure, mpl_figure.Figure):
                if figure._suptitle:
                    return figure._suptitle.get_text()
                if figure.axes:
                    return figure.axes[0].get_title()
            elif isinstance(figure, mpl_axes.Axes):
                return figure.get_title()
            return None
        except Exception:
            return None
    
    def _get_axes(self, figure: Any) -> List[Any]:
        """Get all axes in the seaborn figure."""
        try:
            # Handle seaborn grid objects
            if hasattr(figure, 'axes'):
                axes = figure.axes
                # Check if axes is a numpy array or list
                if hasattr(axes, 'flatten'):
                    return axes.flatten().tolist()
                elif isinstance(axes, list):
                    return axes
                else:
                    return []
            elif hasattr(figure, 'fig'):
                return figure.fig.axes
            elif hasattr(figure, 'figure'):
                return figure.figure.axes
            elif isinstance(figure, mpl_figure.Figure):
                return figure.axes
            elif isinstance(figure, mpl_axes.Axes):
                return [figure]
            else:
                return []
        except Exception:
            return []
    
    def _get_axes_count(self, figure: Any) -> int:
        """Get the number of axes in the seaborn figure."""
        return len(self._get_axes(figure))
    
    def _extract_seaborn_info(self, figure: Any) -> Dict[str, Any]:
        """Extract seaborn-specific information."""
        seaborn_info = {}
        
        try:
            # Detect seaborn plot types
            plot_type = self._detect_seaborn_plot_type(figure)
            seaborn_info["plot_type"] = plot_type
            
            # Extract grid information for FacetGrid, PairGrid, etc.
            if hasattr(figure, 'axes'):
                axes = figure.axes
                # Handle both numpy arrays and lists
                if hasattr(axes, 'shape'):
                    seaborn_info["grid_shape"] = axes.shape
                    seaborn_info["grid_size"] = axes.size
                elif isinstance(axes, list):
                    seaborn_info["grid_shape"] = (len(axes), 1)
                    seaborn_info["grid_size"] = len(axes)
            
            # Extract color palette information
            if hasattr(figure, 'colormap'):
                seaborn_info["colormap"] = str(figure.colormap)
            
            # Extract facet information
            if hasattr(figure, 'col_names'):
                seaborn_info["facet_columns"] = figure.col_names
            if hasattr(figure, 'row_names'):
                seaborn_info["facet_rows"] = figure.row_names
            
            # Extract pair plot information
            if hasattr(figure, 'x_vars'):
                seaborn_info["x_variables"] = figure.x_vars
            if hasattr(figure, 'y_vars'):
                seaborn_info["y_variables"] = figure.y_vars
            
        except Exception as e:
            logger.warning(f"Error extracting seaborn info: {str(e)}")
        
        return seaborn_info
    
    def _detect_seaborn_plot_type(self, figure: Any) -> str:
        """Detect the specific type of seaborn plot."""
        try:
            if hasattr(figure, '__class__'):
                class_name = figure.__class__.__name__
                
                # Common seaborn plot types
                if 'FacetGrid' in class_name:
                    return "facet_grid"
                elif 'PairGrid' in class_name:
                    return "pair_grid"
                elif 'JointGrid' in class_name:
                    return "joint_grid"
                elif 'ClusterGrid' in class_name:
                    return "cluster_grid"
                elif 'Heatmap' in class_name:
                    return "heatmap"
                elif 'ViolinPlot' in class_name:
                    return "violin_plot"
                elif 'BoxPlot' in class_name:
                    return "box_plot"
                elif 'BarPlot' in class_name:
                    return "bar_plot"
                elif 'LinePlot' in class_name:
                    return "line_plot"
                elif 'ScatterPlot' in class_name:
                    return "scatter_plot"
                elif 'HistPlot' in class_name:
                    return "histogram"
                elif 'KdePlot' in class_name:
                    return "kernel_density_estimation"
                elif 'RegPlot' in class_name:
                    return "regression_plot"
                elif 'LmPlot' in class_name:
                    return "linear_model_plot"
                elif 'ResidPlot' in class_name:
                    return "residual_plot"
                elif 'DistPlot' in class_name:
                    return "distribution_plot"
                elif 'JointPlot' in class_name:
                    return "joint_plot"
                elif 'PairPlot' in class_name:
                    return "pair_plot"
                elif 'RelPlot' in class_name:
                    return "relational_plot"
                elif 'CatPlot' in class_name:
                    return "categorical_plot"
                elif 'PointPlot' in class_name:
                    return "point_plot"
                elif 'CountPlot' in class_name:
                    return "count_plot"
                elif 'Stripplot' in class_name:
                    return "strip_plot"
                elif 'SwarmPlot' in class_name:
                    return "swarm_plot"
                else:
                    # Try to detect plot type from axes content
                    return self._detect_plot_type_from_axes(figure)
            
            return "unknown"
        except Exception:
            return "unknown"
    
    def _detect_plot_type_from_axes(self, figure: Any) -> str:
        """Detect plot type by examining the axes content."""
        try:
            axes = self._get_axes(figure)
            if not axes:
                return "unknown"
            
            # Check the first axis for plot characteristics
            ax = axes[0]
            
            # Check for heatmap (has image)
            if hasattr(ax, 'images') and ax.images:
                return "heatmap"
            
            # Check for scatter plot (has collections with offsets)
            if hasattr(ax, 'collections') and ax.collections:
                for collection in ax.collections:
                    if hasattr(collection, 'get_offsets'):
                        offsets = collection.get_offsets()
                        if offsets is not None and len(offsets) > 0:
                            return "scatter_plot"
            
            # Check for line plot
            if hasattr(ax, 'lines') and ax.lines:
                return "line_plot"
            
            # Check for bar plot (has patches)
            if hasattr(ax, 'patches') and ax.patches:
                return "bar_plot"
            
            # Check for histogram (has patches and specific characteristics)
            if hasattr(ax, 'patches') and ax.patches:
                # This is a simplified check - could be enhanced
                return "histogram"
            
            return "unknown_seaborn_plot"
            
        except Exception:
            return "unknown_seaborn_plot"
    
    def _get_data_points(self, figure: Any) -> int:
        """Get the number of data points in the seaborn figure."""
        try:
            total_points = 0
            axes = self._get_axes(figure)
            
            for ax in axes:
                for collection in ax.collections:
                    if hasattr(collection, 'get_offsets'):
                        offsets = collection.get_offsets()
                        if offsets is not None:
                            total_points += len(offsets)
                
                for line in ax.lines:
                    if hasattr(line, 'get_xdata'):
                        xdata = line.get_xdata()
                        if xdata is not None:
                            total_points += len(xdata)
                
                for patch in ax.patches:
                    total_points += 1
            
            return total_points
        except Exception:
            return 0
    
    def _get_data_types(self, figure: Any) -> List[str]:
        """Get the types of data in the seaborn figure."""
        data_types = set()
        
        try:
            axes = self._get_axes(figure)
            
            for ax in axes:
                # Check for different types of plots
                if ax.collections:
                    data_types.add("scatter")
                if ax.lines:
                    data_types.add("line")
                if ax.patches:
                    data_types.add("bar")
                if ax.images:
                    data_types.add("image")
                if ax.texts:
                    data_types.add("text")
            
            # Add seaborn-specific types
            plot_type = self._detect_seaborn_plot_type(figure)
            if plot_type != "unknown":
                data_types.add(plot_type)
            
        except Exception as e:
            logger.warning(f"Error getting data types: {str(e)}")
        
        return list(data_types)
    
    def _get_statistics(self, figure: Any) -> Dict[str, Any]:
        """Get statistical information about the data in the seaborn figure."""
        stats = {}
        
        try:
            axes = self._get_axes(figure)
            
            for i, ax in enumerate(axes):
                ax_stats = {}
                
                # Extract data from collections (scatter plots, etc.)
                for collection in ax.collections:
                    if hasattr(collection, 'get_offsets'):
                        offsets = collection.get_offsets()
                        if offsets is not None and len(offsets) > 0:
                            x_data = offsets[:, 0]
                            y_data = offsets[:, 1]
                            
                            # Check for valid data before calculating statistics
                            if len(x_data) > 0 and len(y_data) > 0:
                                try:
                                    ax_stats[f"collection_{len(ax_stats)}"] = {
                                        "count": len(offsets),
                                        "x_mean": float(np.mean(x_data)),
                                        "x_std": float(np.std(x_data)),
                                        "x_min": float(np.min(x_data)),
                                        "x_max": float(np.max(x_data)),
                                        "y_mean": float(np.mean(y_data)),
                                        "y_std": float(np.std(y_data)),
                                        "y_min": float(np.min(y_data)),
                                        "y_max": float(np.max(y_data)),
                                    }
                                except (ValueError, RuntimeWarning):
                                    # Skip statistics if calculation fails
                                    ax_stats[f"collection_{len(ax_stats)}"] = {
                                        "count": len(offsets),
                                        "error": "Could not calculate statistics"
                                    }
                
                # Extract data from lines
                for line in ax.lines:
                    if hasattr(line, 'get_xdata') and hasattr(line, 'get_ydata'):
                        x_data = line.get_xdata()
                        y_data = line.get_ydata()
                        
                        if x_data is not None and y_data is not None and len(x_data) > 0 and len(y_data) > 0:
                            try:
                                ax_stats[f"line_{len(ax_stats)}"] = {
                                    "count": len(x_data),
                                    "x_mean": float(np.mean(x_data)),
                                    "x_std": float(np.std(x_data)),
                                    "x_min": float(np.min(x_data)),
                                    "x_max": float(np.max(x_data)),
                                    "y_mean": float(np.mean(y_data)),
                                    "y_std": float(np.std(y_data)),
                                    "y_min": float(np.min(y_data)),
                                    "y_max": float(np.max(y_data)),
                                }
                            except (ValueError, RuntimeWarning):
                                # Skip statistics if calculation fails
                                ax_stats[f"line_{len(ax_stats)}"] = {
                                    "count": len(x_data),
                                    "error": "Could not calculate statistics"
                                }
                
                if ax_stats:
                    stats[f"axis_{i}"] = ax_stats
            
        except Exception as e:
            logger.warning(f"Error getting statistics: {str(e)}")
        
        return stats
    
    def _get_colors(self, figure: Any) -> List[str]:
        """Get the colors used in the seaborn figure."""
        colors = set()
        
        try:
            axes = self._get_axes(figure)
            
            for ax in axes:
                # Get colors from collections
                for collection in ax.collections:
                    if hasattr(collection, 'get_facecolor'):
                        facecolor = collection.get_facecolor()
                        if facecolor is not None:
                            if len(facecolor.shape) > 1:
                                # Multiple colors
                                for color in facecolor:
                                    colors.add(to_hex(color))
                            else:
                                colors.add(to_hex(facecolor))
                    
                    if hasattr(collection, 'get_edgecolor'):
                        edgecolor = collection.get_edgecolor()
                        if edgecolor is not None:
                            if len(edgecolor.shape) > 1:
                                for color in edgecolor:
                                    colors.add(to_hex(color))
                            else:
                                colors.add(to_hex(edgecolor))
                
                # Get colors from lines
                for line in ax.lines:
                    color = line.get_color()
                    if color is not None:
                        colors.add(to_hex(color))
                
                # Get colors from patches
                for patch in ax.patches:
                    facecolor = patch.get_facecolor()
                    if facecolor is not None:
                        colors.add(to_hex(facecolor))
                    
                    edgecolor = patch.get_edgecolor()
                    if edgecolor is not None:
                        colors.add(to_hex(edgecolor))
            
        except Exception as e:
            logger.warning(f"Error getting colors: {str(e)}")
        
        return list(colors)
    
    def _get_markers(self, figure: Any) -> List[str]:
        """Get the markers used in the seaborn figure."""
        markers = set()
        
        try:
            axes = self._get_axes(figure)
            
            for ax in axes:
                for line in ax.lines:
                    marker = line.get_marker()
                    if marker is not None and marker != 'None':
                        markers.add(str(marker))
                
                for collection in ax.collections:
                    if hasattr(collection, 'get_paths'):
                        # This might be a scatter plot with markers
                        markers.add("scatter")
            
        except Exception as e:
            logger.warning(f"Error getting markers: {str(e)}")
        
        return list(markers)
    
    def _get_line_styles(self, figure: Any) -> List[str]:
        """Get the line styles used in the seaborn figure."""
        line_styles = set()
        
        try:
            axes = self._get_axes(figure)
            
            for ax in axes:
                for line in ax.lines:
                    linestyle = line.get_linestyle()
                    if linestyle is not None and linestyle != 'None':
                        line_styles.add(str(linestyle))
            
        except Exception as e:
            logger.warning(f"Error getting line styles: {str(e)}")
        
        return list(line_styles)
    
    def _get_background_color(self, figure: Any) -> Optional[str]:
        """Get the background color of the seaborn figure."""
        try:
            if hasattr(figure, 'fig'):
                return to_hex(figure.fig.get_facecolor())
            elif hasattr(figure, 'figure'):
                return to_hex(figure.figure.get_facecolor())
            elif isinstance(figure, mpl_figure.Figure):
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
            # Extract grid layout information
            if hasattr(figure, 'axes'):
                detailed_info["grid_layout"] = {
                    "shape": figure.axes.shape,
                    "size": figure.axes.size,
                    "nrows": figure.axes.shape[0],
                    "ncols": figure.axes.shape[1]
                }
            
            # Extract color palette details
            if hasattr(figure, 'colormap'):
                detailed_info["color_palette"] = {
                    "name": str(figure.colormap),
                    "type": type(figure.colormap).__name__
                }
            
            # Extract facet information in detail
            if hasattr(figure, 'col_names'):
                detailed_info["facet_columns"] = {
                    "names": figure.col_names,
                    "count": len(figure.col_names)
                }
            if hasattr(figure, 'row_names'):
                detailed_info["facet_rows"] = {
                    "names": figure.row_names,
                    "count": len(figure.row_names)
                }
            
        except Exception as e:
            logger.warning(f"Error extracting detailed info: {str(e)}")
        
        return detailed_info 