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
from matplotlib.markers import MarkerStyle
import webcolors

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
        Analyze a matplotlib figure and extract comprehensive information.
        
        Args:
            figure: The matplotlib figure object
            detail_level: Level of detail ("low", "medium", "high")
            include_data: Whether to include data analysis
            include_colors: Whether to include color analysis
            include_statistics: Whether to include statistical analysis
            
        Returns:
            Dictionary containing the analysis results
        """
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
                "figure_type": "matplotlib",
                "figure_info": figure_info,
                "axis_info": axis_info,
                "colors": colors,
                "statistics": statistics
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing matplotlib figure: {str(e)}")
            return {
                "figure_type": "matplotlib",
                "error": str(e),
                "figure_info": {},
                "axis_info": {"axes": [], "figure_title": "", "total_axes": 0},
                "colors": [],
                "statistics": {"per_curve": [], "per_axis": []}
            }
    
    def _get_figure_type(self, figure: Any) -> str:
        """Get the type of the matplotlib figure (standardized)."""
        if isinstance(figure, mpl_figure.Figure):
            return "matplotlib.Figure"
        elif isinstance(figure, mpl_axes.Axes):
            return "matplotlib.Axes"
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
    
    def _get_axis_title(self, ax: Any) -> Optional[str]:
        """Get the title of an individual axis."""
        try:
            return ax.get_title()
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
                # Count data from lines
                for line in ax.lines:
                    if hasattr(line, '_x') and hasattr(line, '_y'):
                        total_points += len(line._x)
                
                # Count data from collections (scatter plots)
                for collection in ax.collections:
                    if hasattr(collection, '_offsets'):
                        total_points += len(collection._offsets)
                
                # Count data from patches (histograms, bar plots)
                for patch in ax.patches:
                    try:
                        if hasattr(patch, 'get_height'):
                            height = patch.get_height()
                            if height > 0:
                                total_points += 1
                    except Exception:
                        continue
                
                # Count data from images
                for image in ax.images:
                    try:
                        if hasattr(image, 'get_array'):
                            img_data = image.get_array()
                            if img_data is not None:
                                total_points += img_data.size
                    except Exception:
                        continue
            
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
                    data_types.append("histogram")
                if ax.images:
                    data_types.append("image")
                if ax.texts:
                    data_types.append("text")
            
            return list(set(data_types))
        except Exception:
            return []
    
    def _get_statistics(self, figure: Any) -> Dict[str, Any]:
        """Get statistical information about the data in the matplotlib figure, per curve and per axis."""
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
                
                # Check for heatmaps first (QuadMesh) - seaborn heatmaps
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
        """Calculate skewness of the data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            skewness = np.mean(((data - mean) / std) ** 3)
            return float(skewness)
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data."""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            kurtosis = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
            return float(kurtosis)
        except Exception:
            return 0.0
    
    def _get_colors(self, figure: Any) -> List[dict]:
        """Get the colors used in the figure, with hex and common name if possible. No colors for heatmaps."""
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
                # NO colors from images (heatmaps)
                # Only extract from lines, collections, patches
                # Colors from lines
                for line in ax.lines:
                    if hasattr(line, '_color'):
                        try:
                            color_hex = to_hex(line._color)
                            color_name = hex_to_name(color_hex)
                            if color_hex not in [c['hex'] for c in colors]:
                                colors.append({"hex": color_hex, "name": color_name})
                        except Exception:
                            continue
                # Colors from collections (scatter plots)
                for collection in ax.collections:
                    if hasattr(collection, '_facecolors'):
                        for color in collection._facecolors:
                            try:
                                hex_color = to_hex(color)
                                color_name = hex_to_name(hex_color)
                                if hex_color not in [c['hex'] for c in colors]:
                                    colors.append({"hex": hex_color, "name": color_name})
                            except Exception:
                                continue
                # Colors from patches (histograms, bar plots)
                for patch in ax.patches:
                    try:
                        if hasattr(patch, 'get_facecolor'):
                            facecolor = patch.get_facecolor()
                            if facecolor is not None:
                                try:
                                    hex_color = to_hex(facecolor)
                                    color_name = hex_to_name(hex_color)
                                    if hex_color not in [c['hex'] for c in colors]:
                                        colors.append({"hex": hex_color, "name": color_name})
                                except Exception:
                                    continue
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Error extracting colors: {str(e)}")
        return colors
    
    def _get_markers(self, figure: Any) -> List[dict]:
        """Get the markers used in the figure, as readable codes and names."""
        markers = []
        try:
            axes = self._get_axes(figure)
            for ax in axes:
                for line in ax.lines:
                    marker_code = line.get_marker() if hasattr(line, 'get_marker') else None
                    if marker_code and marker_code != 'None' and marker_code not in [m['code'] for m in markers]:
                        try:
                            marker_name = MarkerStyle(marker_code).get_marker()
                        except Exception:
                            marker_name = str(marker_code)
                        markers.append({"code": marker_code, "name": marker_name})
        except Exception as e:
            logger.warning(f"Error extracting markers: {str(e)}")
        return markers
    
    def _get_line_styles(self, figure: Any) -> List[dict]:
        """Get the line styles used in the figure, with codes and names."""
        def line_style_to_name(style_code):
            """Convert matplotlib line style code to readable name."""
            style_names = {
                '-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted',
                'None': 'none', ' ': 'none', '': 'none'
            }
            return style_names.get(str(style_code), str(style_code))
        
        styles = []
        try:
            axes = self._get_axes(figure)
            
            for ax in axes:
                for line in ax.lines:
                    if hasattr(line, '_linestyle') and line._linestyle != 'None':
                        style_code = line._linestyle
                        style_name = line_style_to_name(style_code)
                        if style_code not in [s['code'] for s in styles]:
                            styles.append({"code": style_code, "name": style_name})
        
        except Exception as e:
            logger.warning(f"Error extracting line styles: {str(e)}")
        
        return styles
    
    def _get_background_color(self, figure: Any) -> Optional[dict]:
        """Get the background color of the figure, with hex and common name if possible."""
        def hex_to_name(hex_color):
            try:
                import webcolors
                return webcolors.hex_to_name(hex_color)
            except Exception:
                return None
        
        try:
            if isinstance(figure, mpl_figure.Figure):
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

    def _get_figure_info(self, figure: Any) -> Dict[str, Any]:
        """Get basic information about the matplotlib figure."""
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

    def _get_axis_info(self, figure: Any) -> Dict[str, Any]:
        """Get detailed information about axes, including titles and labels."""
        axis_info = {"axes": [], "figure_title": "", "total_axes": 0}
        
        try:
            axes = self._get_axes(figure)
            axis_info["total_axes"] = len(axes)
            
            # Get figure title
            if hasattr(figure, '_suptitle') and figure._suptitle:
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
                    
                    # Check collections (scatter plots, etc.)
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