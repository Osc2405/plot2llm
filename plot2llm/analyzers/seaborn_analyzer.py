"""
Seaborn-specific analyzer for extracting information from seaborn figures.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.figure as mpl_figure
import matplotlib.axes as mpl_axes
from matplotlib.colors import to_hex
import warnings

from .base_analyzer import BaseAnalyzer

# Configure numpy to suppress warnings
np.seterr(all='ignore')  # Suppress all numpy warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

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
        """Analyze a seaborn figure and extract comprehensive information."""
        try:
            # Basic figure information
            figure_info = self._get_figure_info(figure)
            
            # Detailed axis information
            axis_info = self._get_axis_info(figure)
            
            # Color information
            colors = self._get_colors(figure) if include_colors else []
            
            # Statistical information
            statistics = self._get_statistics(figure) if include_statistics else {"per_curve": [], "per_axis": []}
            
            # Combine all information
            result = {
                "figure_type": "seaborn",
                "figure_info": figure_info,
                "axis_info": axis_info,
                "colors": colors,
                "statistics": statistics
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing seaborn figure: {str(e)}")
            return {
                "figure_type": "seaborn",
                "error": str(e),
                "figure_info": {},
                "axis_info": {"axes": [], "figure_title": "", "total_axes": 0},
                "colors": [],
                "statistics": {"per_curve": [], "per_axis": []}
            }
    
    def _get_figure_type(self, figure: Any) -> str:
        """Get the type of the seaborn figure."""
        try:
            # Check for seaborn-specific types
            if hasattr(figure, '__class__'):
                class_name = figure.__class__.__name__
                module_name = figure.__class__.__module__
                
                # Debug logging
                logger.debug(f"SeabornAnalyzer checking figure: class_name={class_name}, module_name={module_name}")
                
                if 'seaborn' in module_name:
                    if 'FacetGrid' in class_name:
                        logger.debug("Detected FacetGrid")
                        return "seaborn.FacetGrid"
                    elif 'PairGrid' in class_name:
                        logger.debug("Detected PairGrid")
                        return "seaborn.PairGrid"
                    elif 'JointGrid' in class_name:
                        logger.debug("Detected JointGrid")
                        return "seaborn.JointGrid"
                    elif 'Heatmap' in class_name:
                        logger.debug("Detected Heatmap")
                        return "seaborn.Heatmap"
                    elif 'ClusterGrid' in class_name:
                        logger.debug("Detected ClusterGrid")
                        return "seaborn.ClusterGrid"
                    else:
                        logger.debug(f"Detected generic seaborn: {class_name}")
                        return f"seaborn.{class_name}"
                
                # Fall back to matplotlib types
                if isinstance(figure, mpl_figure.Figure):
                    logger.debug("Detected matplotlib.Figure")
                    return "matplotlib.Figure"
                elif isinstance(figure, mpl_axes.Axes):
                    logger.debug("Detected matplotlib.Axes")
                    return "matplotlib.Axes"
            
            logger.debug("No specific type detected, returning unknown")
            return "unknown"
        except Exception as e:
            logger.warning(f"Error in _get_figure_type: {str(e)}")
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
    
    def _get_figure_info(self, figure: Any) -> Dict[str, Any]:
        """Get basic information about the seaborn figure."""
        try:
            figure_type = self._get_figure_type(figure)
            dimensions = self._get_dimensions(figure)
            title = self._get_title(figure)
            axes_count = self._get_axes_count(figure)
            
            return {
                "figure_type": figure_type,
                "dimensions": [float(dim) for dim in dimensions],
                "title": title,
                "axes_count": axes_count
            }
        except Exception as e:
            logger.warning(f"Error getting figure info: {str(e)}")
            return {
                "figure_type": "unknown",
                "dimensions": [0, 0],
                "title": None,
                "axes_count": 0
            }
    
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
        """Detect seaborn plot type from the figure object."""
        try:
            if hasattr(figure, '__class__'):
                class_name = figure.__class__.__name__
                module_name = figure.__class__.__module__
                if 'seaborn' in module_name:
                    if 'FacetGrid' in class_name:
                        return "FacetGrid"
                    elif 'PairGrid' in class_name:
                        return "PairGrid"
                    elif 'JointGrid' in class_name:
                        return "JointGrid"
                    elif 'Heatmap' in class_name:
                        return "heatmap"
                    elif 'ClusterGrid' in class_name:
                        return "clustermap"
                    else:
                        return class_name
            # Fallback: try to detect from axes
            return self._detect_plot_type_from_axes(figure)
        except Exception:
            return "unknown"
    
    def _detect_plot_type_from_axes(self, figure: Any) -> str:
        """Detect plot type by examining the axes content."""
        try:
            axes = self._get_axes(figure)
            if not axes:
                return "unknown"
            ax = axes[0]
            # Check for heatmap (QuadMesh)
            for collection in ax.collections:
                if collection.__class__.__name__ == "QuadMesh" and hasattr(collection, 'get_array') and collection.get_array() is not None:
                    return "heatmap"
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
                # Count data from images (heatmaps)
                for image in ax.images:
                    try:
                        if hasattr(image, 'get_array'):
                            img_data = image.get_array()
                            if img_data is not None:
                                total_points += img_data.size
                    except Exception:
                        continue
                
                # Count data from collections (scatter plots, etc.)
                for collection in ax.collections:
                    if hasattr(collection, 'get_offsets'):
                        offsets = collection.get_offsets()
                        if offsets is not None:
                            total_points += len(offsets)
                
                # Count data from lines
                for line in ax.lines:
                    if hasattr(line, 'get_xdata'):
                        xdata = line.get_xdata()
                        if xdata is not None:
                            total_points += len(xdata)
                
                # Count data from patches (histograms, bar plots)
                for patch in ax.patches:
                    total_points += 1
            
            return total_points
        except Exception:
            return 0
    
    def _get_data_types(self, figure: Any) -> List[str]:
        data_types = set()
        try:
            axes = self._get_axes(figure)
            for ax in axes:
                # Check for heatmaps first (QuadMesh)
                is_heatmap = False
                for collection in ax.collections:
                    if collection.__class__.__name__ == "QuadMesh" and hasattr(collection, 'get_array') and collection.get_array() is not None:
                        data_types.add("heatmap")
                        is_heatmap = True
                        break
                if not is_heatmap and hasattr(ax, 'images') and ax.images:
                    for image in ax.images:
                        if hasattr(image, 'get_array') and image.get_array() is not None:
                            data_types.add("heatmap")
                            is_heatmap = True
                            break
                if is_heatmap:
                    continue  # Skip other types if heatmap
                # Check for different types of plots (only if not a heatmap)
                if ax.collections:
                    for collection in ax.collections:
                        if hasattr(collection, 'get_offsets'):
                            offsets = collection.get_offsets()
                            if offsets is not None and len(offsets) > 0:
                                data_types.add("scatter_plot")
                                break
                if ax.lines:
                    data_types.add("line_plot")
                if ax.patches:
                    data_types.add("histogram")
                if ax.texts:
                    if "heatmap" not in data_types:
                        data_types.add("text")
            plot_type = self._detect_seaborn_plot_type(figure)
            if plot_type != "unknown":
                data_types.add(plot_type)
        except Exception as e:
            logger.warning(f"Error getting data types: {str(e)}")
        return list(data_types)
    
    def _get_statistics(self, figure: Any) -> Dict[str, Any]:
        """Get statistical information about the data in the seaborn figure, per curve and per axis."""
        stats = {"per_curve": [], "per_axis": []}
        
        try:
            axes = self._get_axes(figure)
            all_data = []
            
            # Statistics per axis/subplot
            for i, ax in enumerate(axes):
                axis_data = []
                axis_stats = {
                    "axis_index": i,
                    "title": ax.get_title() if ax.get_title() else f"Subplot {i+1}",
                    "data_types": [],
                    "data_points": 0,
                    "matrix_data": None
                }
                
                # Check for heatmaps first (QuadMesh)
                has_heatmap = False
                for collection in ax.collections:
                    if collection.__class__.__name__ == "QuadMesh" and hasattr(collection, 'get_array'):
                        try:
                            arr = collection.get_array()
                            if arr is not None:
                                arr = np.asarray(arr)
                                if hasattr(arr, 'mask'):
                                    arr = np.ma.filled(arr, np.nan)
                                matrix_data = arr.tolist() if hasattr(arr, 'tolist') else arr
                                axis_stats["matrix_data"] = {
                                    "shape": arr.shape if hasattr(arr, 'shape') else (arr,),
                                    "data": matrix_data,
                                    "min_value": float(np.nanmin(arr)),
                                    "max_value": float(np.nanmax(arr)),
                                    "mean_value": float(np.nanmean(arr)),
                                    "std_value": float(np.nanstd(arr))
                                }
                                flat_data = arr.flatten() if hasattr(arr, 'flatten') else arr
                                axis_data.extend(flat_data)
                                axis_stats["data_points"] += len(flat_data)
                                axis_stats["data_types"].append("heatmap")
                                has_heatmap = True
                                break
                        except Exception as e:
                            logger.warning(f"Error processing QuadMesh: {str(e)}")
                            continue
                
                # If no QuadMesh, try images
                if not has_heatmap:
                    for image in ax.images:
                        try:
                            if hasattr(image, 'get_array'):
                                img_data = image.get_array()
                                if img_data is not None:
                                    img_data = np.asarray(img_data)
                                    if hasattr(img_data, 'mask'):
                                        img_data = np.ma.filled(img_data, np.nan)
                                    matrix_data = img_data.tolist() if hasattr(img_data, 'tolist') else img_data
                                    axis_stats["matrix_data"] = {
                                        "shape": img_data.shape if hasattr(img_data, 'shape') else (img_data,),
                                        "data": matrix_data,
                                        "min_value": float(np.nanmin(img_data)),
                                        "max_value": float(np.nanmax(img_data)),
                                        "mean_value": float(np.nanmean(img_data)),
                                        "std_value": float(np.nanstd(img_data))
                                    }
                                    flat_data = img_data.flatten() if hasattr(img_data, 'flatten') else img_data
                                    axis_data.extend(flat_data)
                                    axis_stats["data_points"] += len(flat_data)
                                    axis_stats["data_types"].append("heatmap")
                                    has_heatmap = True
                                    break
                        except Exception as e:
                            logger.warning(f"Error processing heatmap image: {str(e)}")
                            continue
                
                # If this is a heatmap, skip other data types
                if has_heatmap:
                    if axis_data and len(axis_data) > 0:
                        try:
                            axis_data = np.array(axis_data)
                            # Remove any NaN or infinite values
                            axis_data = axis_data[np.isfinite(axis_data)]
                            
                            if len(axis_data) > 0:
                                axis_stats.update({
                                    "mean": float(np.nanmean(axis_data)),
                                    "std": float(np.nanstd(axis_data)),
                                    "min": float(np.nanmin(axis_data)),
                                    "max": float(np.nanmax(axis_data)),
                                    "median": float(np.nanmedian(axis_data)),
                                    "outliers": [],
                                    "local_var": float(np.nanvar(axis_data[:max(1, len(axis_data)//10)])),
                                    "trend": 0.0,
                                    "skewness": 0.0,
                                    "kurtosis": 0.0
                                })
                                
                                # Calculate outliers only if we have enough data
                                if len(axis_data) > 3:
                                    mean_val = np.nanmean(axis_data)
                                    std_val = np.nanstd(axis_data)
                                    if std_val > 0:
                                        outliers = axis_data[np.abs(axis_data - mean_val) > 2 * std_val]
                                        axis_stats["outliers"] = [float(val) for val in outliers]
                                
                                # Calculate trend only if we have enough data
                                if len(axis_data) > 1:
                                    try:
                                        trend = np.polyfit(np.arange(len(axis_data)), axis_data, 1)[0]
                                        axis_stats["trend"] = float(trend)
                                    except Exception:
                                        axis_stats["trend"] = 0.0
                                
                                # Calculate skewness and kurtosis only if we have enough data
                                if len(axis_data) > 2:
                                    try:
                                        axis_stats["skewness"] = float(self._calculate_skewness(axis_data))
                                        axis_stats["kurtosis"] = float(self._calculate_kurtosis(axis_data))
                                    except Exception:
                                        axis_stats["skewness"] = 0.0
                                        axis_stats["kurtosis"] = 0.0
                                
                                all_data.extend(axis_data)
                            else:
                                axis_stats.update({
                                    "mean": None, "std": None, "min": None, "max": None, "median": None,
                                    "outliers": [], "local_var": None, "trend": None,
                                    "skewness": None, "kurtosis": None
                                })
                        except Exception as e:
                            logger.warning(f"Error calculating heatmap statistics for axis {i}: {str(e)}")
                            axis_stats.update({
                                "mean": None, "std": None, "min": None, "max": None, "median": None,
                                "outliers": [], "local_var": None, "trend": None,
                                "skewness": None, "kurtosis": None
                            })
                    else:
                        axis_stats.update({
                            "mean": None, "std": None, "min": None, "max": None, "median": None,
                            "outliers": [], "local_var": None, "trend": None,
                            "skewness": None, "kurtosis": None
                        })
                    stats["per_axis"].append(axis_stats)
                    continue
                
                # Collect data from collections (scatter plots, etc.)
                for collection in ax.collections:
                    if hasattr(collection, 'get_offsets'):
                        offsets = collection.get_offsets()
                        if offsets is not None and len(offsets) > 0:
                            x_data = offsets[:, 0]
                            y_data = offsets[:, 1]
                            
                            # Add both x and y data
                            axis_data.extend(x_data)
                            axis_data.extend(y_data)
                            axis_stats["data_points"] += len(x_data) + len(y_data)
                            if "scatter_plot" not in axis_stats["data_types"]:
                                axis_stats["data_types"].append("scatter_plot")
                
                # Collect data from lines
                for line in ax.lines:
                    if hasattr(line, 'get_xdata') and hasattr(line, 'get_ydata'):
                        x_data = line.get_xdata()
                        y_data = line.get_ydata()
                        
                        if x_data is not None and y_data is not None and len(x_data) > 0 and len(y_data) > 0:
                            axis_data.extend(x_data)
                            axis_data.extend(y_data)
                            axis_stats["data_points"] += len(x_data) + len(y_data)
                            if "line_plot" not in axis_stats["data_types"]:
                                axis_stats["data_types"].append("line_plot")
                
                # Collect data from patches (histograms, bar plots)
                for patch in ax.patches:
                    try:
                        if hasattr(patch, 'get_height'):
                            height = patch.get_height()
                            if height > 0:
                                axis_data.append(float(height))
                                axis_stats["data_points"] += 1
                                if "histogram" not in axis_stats["data_types"]:
                                    axis_stats["data_types"].append("histogram")
                    except Exception:
                        continue
                
                # Calculate statistics for this axis if it has data
                if axis_data and len(axis_data) > 0:
                    try:
                        axis_data = np.array(axis_data)
                        # Remove any NaN or infinite values
                        axis_data = axis_data[np.isfinite(axis_data)]
                        
                        if len(axis_data) > 0:
                            axis_stats.update({
                                "mean": float(np.nanmean(axis_data)),
                                "std": float(np.nanstd(axis_data)),
                                "min": float(np.nanmin(axis_data)),
                                "max": float(np.nanmax(axis_data)),
                                "median": float(np.nanmedian(axis_data)),
                                "outliers": [],
                                "local_var": float(np.nanvar(axis_data[:max(1, len(axis_data)//10)])),
                                "trend": 0.0,
                                "skewness": 0.0,
                                "kurtosis": 0.0
                            })
                            
                            # Calculate outliers only if we have enough data
                            if len(axis_data) > 3:
                                mean_val = np.nanmean(axis_data)
                                std_val = np.nanstd(axis_data)
                                if std_val > 0:
                                    outliers = axis_data[np.abs(axis_data - mean_val) > 2 * std_val]
                                    axis_stats["outliers"] = [float(val) for val in outliers]
                            
                            # Calculate trend only if we have enough data
                            if len(axis_data) > 1:
                                try:
                                    trend = np.polyfit(np.arange(len(axis_data)), axis_data, 1)[0]
                                    axis_stats["trend"] = float(trend)
                                except Exception:
                                    axis_stats["trend"] = 0.0
                            
                            # Calculate skewness and kurtosis only if we have enough data
                            if len(axis_data) > 2:
                                try:
                                    axis_stats["skewness"] = float(self._calculate_skewness(axis_data))
                                    axis_stats["kurtosis"] = float(self._calculate_kurtosis(axis_data))
                                except Exception:
                                    axis_stats["skewness"] = 0.0
                                    axis_stats["kurtosis"] = 0.0
                            
                            all_data.extend(axis_data)
                        else:
                            # No valid data
                            axis_stats.update({
                                "mean": None, "std": None, "min": None, "max": None, "median": None,
                                "outliers": [], "local_var": None, "trend": None,
                                "skewness": None, "kurtosis": None
                            })
                    except Exception as e:
                        logger.warning(f"Error calculating statistics for axis {i}: {str(e)}")
                        axis_stats.update({
                            "mean": None, "std": None, "min": None, "max": None, "median": None,
                            "outliers": [], "local_var": None, "trend": None,
                            "skewness": None, "kurtosis": None
                        })
                else:
                    # No data
                    axis_stats.update({
                        "mean": None, "std": None, "min": None, "max": None, "median": None,
                        "outliers": [], "local_var": None, "trend": None,
                        "skewness": None, "kurtosis": None
                    })
                
                stats["per_axis"].append(axis_stats)
            
            # Statistics per curve (for line plots)
            for ax in axes:
                for line in ax.lines:
                    if hasattr(line, 'get_xdata') and hasattr(line, 'get_ydata'):
                        x_data = line.get_xdata()
                        y_data = line.get_ydata()
                        
                        if x_data is not None and y_data is not None and len(x_data) > 0 and len(y_data) > 0:
                            try:
                                # Remove any NaN or infinite values
                                x_data = np.array(x_data)[np.isfinite(x_data)]
                                y_data = np.array(y_data)[np.isfinite(y_data)]
                                
                                if len(x_data) > 0 and len(y_data) > 0:
                                    curve_stats = {
                                        "label": line.get_label(),
                                        "x_mean": float(np.nanmean(x_data)),
                                        "x_std": float(np.nanstd(x_data)),
                                        "x_min": float(np.nanmin(x_data)),
                                        "x_max": float(np.nanmax(x_data)),
                                        "y_mean": float(np.nanmean(y_data)),
                                        "y_std": float(np.nanstd(y_data)),
                                        "y_min": float(np.nanmin(y_data)),
                                        "y_max": float(np.nanmax(y_data)),
                                    }
                                    stats["per_curve"].append(curve_stats)
                            except Exception as e:
                                logger.warning(f"Error calculating curve statistics: {str(e)}")
                                continue
            
            # Global statistics
            if all_data and len(all_data) > 0:
                try:
                    all_data = np.array(all_data)
                    all_data = all_data[np.isfinite(all_data)]
                    
                    if len(all_data) > 0:
                        stats["global"] = {
                            "mean": float(np.nanmean(all_data)),
                            "std": float(np.nanstd(all_data)),
                            "min": float(np.nanmin(all_data)),
                            "max": float(np.nanmax(all_data)),
                            "median": float(np.nanmedian(all_data)),
                        }
                except Exception as e:
                    logger.warning(f"Error calculating global statistics: {str(e)}")
            
        except Exception as e:
            logger.warning(f"Error getting statistics: {str(e)}")
        
        return stats
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness with proper handling of edge cases."""
        try:
            if len(data) < 3:
                return 0.0
            
            # Remove NaN and infinite values
            clean_data = data[np.isfinite(data)]
            if len(clean_data) < 3:
                return 0.0
            
            mean_val = np.nanmean(clean_data)
            std_val = np.nanstd(clean_data)
            
            if std_val == 0 or not np.isfinite(std_val):
                return 0.0
            
            # Calculate skewness
            skewness = np.nanmean(((clean_data - mean_val) / std_val) ** 3)
            return float(skewness) if np.isfinite(skewness) else 0.0
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis with proper handling of edge cases."""
        try:
            if len(data) < 4:
                return 0.0
            
            # Remove NaN and infinite values
            clean_data = data[np.isfinite(data)]
            if len(clean_data) < 4:
                return 0.0
            
            mean_val = np.nanmean(clean_data)
            std_val = np.nanstd(clean_data)
            
            if std_val == 0 or not np.isfinite(std_val):
                return 0.0
            
            # Calculate kurtosis
            kurtosis = np.nanmean(((clean_data - mean_val) / std_val) ** 4) - 3
            return float(kurtosis) if np.isfinite(kurtosis) else 0.0
        except Exception:
            return 0.0
    
    def _get_colors(self, figure: Any) -> List[dict]:
        """Get the colors used in the seaborn figure, with hex and common name if possible. No colors for heatmaps (QuadMesh)."""
        def hex_to_name(hex_color):
            try:
                import webcolors
                return webcolors.hex_to_name(hex_color)
            except Exception:
                return None
        colors = []
        try:
            axes = self._get_axes(figure)
            for ax in axes:
                # Skip axis if it has a heatmap (QuadMesh)
                has_heatmap = False
                for collection in ax.collections:
                    if collection.__class__.__name__ == "QuadMesh" and hasattr(collection, 'get_array') and collection.get_array() is not None:
                        has_heatmap = True
                        break
                if not has_heatmap and hasattr(ax, 'images') and ax.images:
                    for image in ax.images:
                        if hasattr(image, 'get_array') and image.get_array() is not None:
                            has_heatmap = True
                            break
                if has_heatmap:
                    continue
                # Colors from collections (not QuadMesh)
                for collection in ax.collections:
                    if collection.__class__.__name__ == "QuadMesh":
                        continue
                    if hasattr(collection, 'get_facecolor'):
                        facecolor = collection.get_facecolor()
                        if facecolor is not None:
                            if len(facecolor.shape) > 1:
                                for color in facecolor:
                                    try:
                                        hex_color = to_hex(color)
                                        color_name = hex_to_name(hex_color)
                                        if hex_color not in [c['hex'] for c in colors]:
                                            colors.append({"hex": hex_color, "name": color_name})
                                    except Exception:
                                        continue
                            else:
                                try:
                                    hex_color = to_hex(facecolor)
                                    color_name = hex_to_name(hex_color)
                                    if hex_color not in [c['hex'] for c in colors]:
                                        colors.append({"hex": hex_color, "name": color_name})
                                except Exception:
                                    continue
                    if hasattr(collection, 'get_edgecolor'):
                        edgecolor = collection.get_edgecolor()
                        if edgecolor is not None:
                            if len(edgecolor.shape) > 1:
                                for color in edgecolor:
                                    try:
                                        hex_color = to_hex(color)
                                        color_name = hex_to_name(hex_color)
                                        if hex_color not in [c['hex'] for c in colors]:
                                            colors.append({"hex": hex_color, "name": color_name})
                                    except Exception:
                                        continue
                            else:
                                try:
                                    hex_color = to_hex(edgecolor)
                                    color_name = hex_to_name(hex_color)
                                    if hex_color not in [c['hex'] for c in colors]:
                                        colors.append({"hex": hex_color, "name": color_name})
                                except Exception:
                                    continue
                # Colors from lines
                for line in ax.lines:
                    color = line.get_color()
                    if color is not None:
                        try:
                            hex_color = to_hex(color)
                            color_name = hex_to_name(hex_color)
                            if hex_color not in [c['hex'] for c in colors]:
                                colors.append({"hex": hex_color, "name": color_name})
                        except Exception:
                            continue
                # Colors from patches
                for patch in ax.patches:
                    facecolor = patch.get_facecolor()
                    if facecolor is not None:
                        try:
                            hex_color = to_hex(facecolor)
                            color_name = hex_to_name(hex_color)
                            if hex_color not in [c['hex'] for c in colors]:
                                colors.append({"hex": hex_color, "name": color_name})
                        except Exception:
                            continue
                    edgecolor = patch.get_edgecolor()
                    if edgecolor is not None:
                        try:
                            hex_color = to_hex(edgecolor)
                            color_name = hex_to_name(hex_color)
                            if hex_color not in [c['hex'] for c in colors]:
                                colors.append({"hex": hex_color, "name": color_name})
                        except Exception:
                            continue
        except Exception as e:
            logger.warning(f"Error getting colors: {str(e)}")
        return colors
    
    def _get_markers(self, figure: Any) -> List[dict]:
        """Get the markers used in the seaborn figure, with codes and names."""
        def marker_code_to_name(marker_code):
            """Convert matplotlib marker code to readable name."""
            marker_names = {
                'o': 'circle', 's': 'square', '^': 'triangle_up', 'v': 'triangle_down',
                'D': 'diamond', 'p': 'plus', '*': 'star', 'h': 'hexagon1', 'H': 'hexagon2',
                'd': 'thin_diamond', '|': 'vline', '_': 'hline', 'P': 'plus_filled',
                'X': 'x_filled', 'x': 'x', '+': 'plus', '1': 'tri_down', '2': 'tri_up',
                '3': 'tri_left', '4': 'tri_right', '8': 'octagon', 'None': 'none'
            }
            return marker_names.get(str(marker_code), str(marker_code))
        
        markers = []
        try:
            axes = self._get_axes(figure)
            
            for ax in axes:
                for line in ax.lines:
                    marker = line.get_marker()
                    if marker is not None and marker != 'None':
                        marker_code = str(marker)
                        marker_name = marker_code_to_name(marker)
                        if marker_code not in [m['code'] for m in markers]:
                            markers.append({"code": marker_code, "name": marker_name})
                
                for collection in ax.collections:
                    if hasattr(collection, 'get_paths'):
                        # This might be a scatter plot with markers
                        if "scatter" not in [m['code'] for m in markers]:
                            markers.append({"code": "scatter", "name": "scatter_points"})
            
        except Exception as e:
            logger.warning(f"Error getting markers: {str(e)}")
        
        return markers
    
    def _get_line_styles(self, figure: Any) -> List[dict]:
        """Get the line styles used in the seaborn figure, with codes and names."""
        def line_style_to_name(style_code):
            """Convert matplotlib line style code to readable name."""
            style_names = {
                '-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted',
                'None': 'none', ' ': 'none', '': 'none'
            }
            return style_names.get(str(style_code), str(style_code))
        
        line_styles = []
        try:
            axes = self._get_axes(figure)
            
            for ax in axes:
                for line in ax.lines:
                    linestyle = line.get_linestyle()
                    if linestyle is not None and linestyle != 'None':
                        style_code = str(linestyle)
                        style_name = line_style_to_name(linestyle)
                        if style_code not in [s['code'] for s in line_styles]:
                            line_styles.append({"code": style_code, "name": style_name})
            
        except Exception as e:
            logger.warning(f"Error getting line styles: {str(e)}")
        
        return line_styles
    
    def _get_background_color(self, figure: Any) -> Optional[dict]:
        """Get the background color of the seaborn figure, with hex and common name if possible."""
        def hex_to_name(hex_color):
            try:
                import webcolors
                return webcolors.hex_to_name(hex_color)
            except Exception:
                return None
        
        try:
            if hasattr(figure, 'fig'):
                bg_color = figure.fig.get_facecolor()
            elif hasattr(figure, 'figure'):
                bg_color = figure.figure.get_facecolor()
            elif isinstance(figure, mpl_figure.Figure):
                bg_color = figure.get_facecolor()
            elif isinstance(figure, mpl_axes.Axes):
                bg_color = figure.get_facecolor()
            else:
                return None
            
            hex_color = to_hex(bg_color)
            color_name = hex_to_name(hex_color)
            return {"hex": hex_color, "name": color_name}
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
    
    def _get_axis_info(self, figure: Any) -> Dict[str, Any]:
        """Get detailed information about axes, including titles and labels."""
        axis_info = {"axes": [], "figure_title": "", "total_axes": 0}
        
        try:
            axes = self._get_axes(figure)
            axis_info["total_axes"] = len(axes)
            
            # Get figure title
            if hasattr(figure, 'suptitle') and figure._suptitle:
                axis_info["figure_title"] = figure._suptitle.get_text()
            elif hasattr(figure, 'get_suptitle'):
                axis_info["figure_title"] = figure.get_suptitle()
            
            for i, ax in enumerate(axes):
                ax_info = {
                    "index": i,
                    "title": "",
                    "x_label": "",
                    "y_label": "",
                    "x_lim": None,
                    "y_lim": None,
                    "has_data": False
                }
                
                # Extract axis title (subplot title)
                try:
                    if hasattr(ax, 'get_title'):
                        title = ax.get_title()
                        if title and title.strip():
                            ax_info["title"] = title.strip()
                except Exception:
                    pass
                
                # Extract X and Y axis labels
                try:
                    if hasattr(ax, 'get_xlabel'):
                        x_label = ax.get_xlabel()
                        if x_label and x_label.strip():
                            ax_info["x_label"] = x_label.strip()
                except Exception:
                    pass
                
                try:
                    if hasattr(ax, 'get_ylabel'):
                        y_label = ax.get_ylabel()
                        if y_label and y_label.strip():
                            ax_info["y_label"] = y_label.strip()
                except Exception:
                    pass
                
                # Extract axis limits
                try:
                    if hasattr(ax, 'get_xlim'):
                        x_lim = ax.get_xlim()
                        if x_lim and len(x_lim) == 2:
                            ax_info["x_lim"] = [float(x_lim[0]), float(x_lim[1])]
                except Exception:
                    pass
                
                try:
                    if hasattr(ax, 'get_ylim'):
                        y_lim = ax.get_ylim()
                        if y_lim and len(y_lim) == 2:
                            ax_info["y_lim"] = [float(y_lim[0]), float(y_lim[1])]
                except Exception:
                    pass
                
                # Check if axis has data
                try:
                    has_data = False
                    
                    # Check collections (scatter plots, heatmaps, etc.)
                    if hasattr(ax, 'collections') and ax.collections:
                        has_data = True
                    
                    # Check lines
                    if hasattr(ax, 'lines') and ax.lines:
                        has_data = True
                    
                    # Check patches (histograms, bar plots)
                    if hasattr(ax, 'patches') and ax.patches:
                        has_data = True
                    
                    # Check images (heatmaps)
                    if hasattr(ax, 'images') and ax.images:
                        has_data = True
                    
                    ax_info["has_data"] = has_data
                except Exception:
                    ax_info["has_data"] = False
                
                axis_info["axes"].append(ax_info)
            
        except Exception as e:
            logger.warning(f"Error getting axis info: {str(e)}")
        
        return axis_info 