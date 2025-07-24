"""
Matplotlib-specific analyzer for extracting information from matplotlib figures.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import matplotlib.patches as mpl_patches
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.markers import MarkerStyle

from plot2llm.utils import serialize_axis_values
from plot2llm.utils_matplotlib import extract_axes_section

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

    def analyze(
        self,
        figure: Any,
        detail_level: str = "medium",
        include_data: bool = True,
        include_colors: bool = True,
        include_statistics: bool = True,
    ) -> dict:
        """
        Analyze a matplotlib figure and extract comprehensive information.
        Returns a dict compatible with tests and formatters.
        """
        import matplotlib.axes as mpl_axes
        import matplotlib.figure as mpl_figure

        if figure is None:
            raise ValueError("Invalid figure object: None")
        if not (
            isinstance(figure, mpl_figure.Figure) or isinstance(figure, mpl_axes.Axes)
        ):
            raise ValueError("Not a matplotlib figure")
        try:
            # Basic info
            figure_info = self._get_figure_info(figure)
            
            # Extract axes information using the new utility
            axes_list = []
            real_axes = self._get_axes(figure)
            for ax in real_axes:
                axes_section = extract_axes_section(ax)
                axes_list.append(axes_section)
            
            # Get additional information
            colors = self._get_colors(figure) if include_colors else []
            statistics = (
                self._get_statistics(figure)
                if include_statistics
                else {"per_curve": [], "per_axis": []}
            )

            # Compose the final output
            return {
                "figure": figure_info,
                "axes": axes_list,
                "colors": colors,
                "statistics": statistics
            }

        except Exception as e:
            logger.error(f"Error analyzing figure: {str(e)}")
            raise

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
            if hasattr(ax, "get_xscale"):
                xscale = ax.get_xscale()
                yscale = ax.get_yscale()
                if xscale == "log" or yscale == "log":
                    return "log"
                elif xscale == "symlog" or yscale == "symlog":
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
                    if hasattr(line, "_x") and hasattr(line, "_y"):
                        total_points += len(line._x)

                # Count data from collections (scatter plots)
                for collection in ax.collections:
                    if hasattr(collection, "_offsets"):
                        total_points += len(collection._offsets)

                # Count data from patches (histograms, bar plots)
                for patch in ax.patches:
                    try:
                        if hasattr(patch, "get_height"):
                            height = patch.get_height()
                            if height > 0:
                                total_points += 1
                    except Exception:
                        continue

                # Count data from images
                for image in ax.images:
                    try:
                        if hasattr(image, "get_array"):
                            img_data = image.get_array()
                            if img_data is not None:
                                total_points += img_data.size
                    except Exception:
                        continue

            return total_points
        except Exception:
            return 0

    def _get_data_types(self, figure: Any) -> List[str]:
        """Get the types of data visualizations present in the figure."""
        data_types = []
        try:
            axes = self._get_axes(figure)

            for ax in axes:
                # Detectar gráficos de barras
                if hasattr(ax, "patches") and any(
                    isinstance(p, matplotlib.patches.Rectangle) and p.get_y() == 0 and p.get_height() > 0
                    for p in ax.patches
                ):
                    data_types.append("bar_plot")
                
                # Detectar gráficos de línea
                if hasattr(ax, "lines") and ax.lines:
                    data_types.append("line_plot")
                
                # Detectar scatter plots
                if hasattr(ax, "collections"):
                    for collection in ax.collections:
                        if isinstance(collection, matplotlib.collections.PathCollection):
                            data_types.append("scatter_plot")
                            break

            return list(set(data_types))
        except Exception:
            return []

    def _detect_axis_type_and_labels(
        self, ax: Any, axis: str = "x"
    ) -> Tuple[str, List[str]]:
        """
        Detecta el tipo de eje y sus etiquetas de forma robusta.

        Args:
            ax: El eje de matplotlib
            axis: 'x' o 'y' para indicar qué eje analizar

        Returns:
            Tuple[str, List[str]]: (tipo_de_eje, lista_de_etiquetas)
        """
        try:
            import pandas as pd

            # Asegurar que los labels estén disponibles
            if hasattr(ax, "figure") and hasattr(ax.figure, "canvas"):
                try:
                    ax.figure.canvas.draw()
                except Exception:
                    pass

            # 1. Obtener etiquetas del eje
            if axis == "x":
                labels = [lbl.get_text().strip() for lbl in ax.get_xticklabels()]
            else:
                labels = [lbl.get_text().strip() for lbl in ax.get_yticklabels()]

            labels = [lbl for lbl in labels if lbl]  # Filtrar etiquetas vacías

            # 2. Buscar en los datos originales
            data_values = []

            # Revisar líneas
            for line in ax.lines:
                if axis == "x" and hasattr(line, "get_xdata"):
                    data = line.get_xdata()
                    if len(data) > 0:
                        data_values.extend(data)
                elif axis == "y" and hasattr(line, "get_ydata"):
                    data = line.get_ydata()
                    if len(data) > 0:
                        data_values.extend(data)

            # Revisar colecciones (scatter plots)
            for collection in ax.collections:
                if hasattr(collection, "get_offsets"):
                    offsets = collection.get_offsets()
                    if offsets is not None and len(offsets) > 0:
                        data_values.extend(offsets[:, 0 if axis == "x" else 1])

            # Revisar patches (barras)
            if axis == "x":
                for patch in ax.patches:
                    if hasattr(patch, "get_x"):
                        data_values.append(patch.get_x())
            else:
                for patch in ax.patches:
                    if hasattr(patch, "get_height"):
                        data_values.append(patch.get_height())

            # 3. Analizar los datos para determinar el tipo
            if len(data_values) > 0:
                data_array = np.array(data_values)

                # Verificar si son fechas
                if np.issubdtype(data_array.dtype, np.datetime64):
                    return "date", [
                        pd.Timestamp(x).strftime("%Y-%m-%d") for x in data_values
                    ]

                # Verificar si son períodos
                if hasattr(data_array, "dtype") and str(data_array.dtype).startswith(
                    "period"
                ):
                    return "period", [str(x) for x in data_values]

                # Verificar si son strings/categorías
                if all(isinstance(x, str) for x in data_values):
                    return "category", data_values

                # Si hay etiquetas no numéricas y coinciden con la cantidad de datos
                if labels and len(labels) == len(set(data_values)):
                    try:
                        float("".join(labels))  # Intentar convertir a número
                    except ValueError:
                        # Si no se puede convertir a número, son categorías
                        return "category", labels

                # Si los valores son enteros consecutivos y hay etiquetas significativas
                if all(
                    isinstance(x, (int, np.integer))
                    or (isinstance(x, float) and x.is_integer())
                    for x in data_values
                ):
                    unique_values = sorted(set(data_values))
                    if len(unique_values) > 1 and unique_values == list(
                        range(int(min(unique_values)), int(max(unique_values)) + 1)
                    ):
                        if labels and len(labels) >= len(unique_values):
                            try:
                                float("".join(labels[: len(unique_values)]))
                            except ValueError:
                                return "category", labels[: len(unique_values)]

                # Si no es ninguno de los anteriores, es numérico
                return "numeric", [str(x) for x in data_values]

            # Si no hay datos pero hay etiquetas no numéricas
            if labels:
                try:
                    float("".join(labels))
                    return "numeric", labels
                except ValueError:
                    return "category", labels

            return "numeric", []

        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Error detecting axis type: {str(e)}")
            return "numeric", []

    def _get_statistics(self, figure: Any) -> Dict[str, Any]:
        """Get statistical information about the data in the figure."""
        def to_native_type(value):
            """Convierte valores NumPy a tipos Python nativos."""
            if isinstance(value, (np.integer, np.floating)):
                return float(value)
            if isinstance(value, (list, tuple)):
                return [to_native_type(v) for v in value]
            if isinstance(value, dict):
                return {k: to_native_type(v) for k, v in value.items()}
            return value

        statistics = {"per_curve": [], "per_axis": []}
        axes = self._get_axes(figure)

        for axis_index, ax in enumerate(axes):
            # Detectar tipo de eje y etiquetas para x/y
            x_type, x_labels = self._detect_axis_type_and_labels(ax, "x")
            y_type, y_labels = self._detect_axis_type_and_labels(ax, "y")
            
            # Obtener tipos de datos presentes en el eje
            data_types = []
            
            # Detectar gráficos de barras
            if hasattr(ax, "patches") and any(
                isinstance(p, mpl_patches.Rectangle) and p.get_y() == 0 and p.get_height() > 0
                for p in ax.patches
            ):
                data_types.append("bar_plot")
            
            axis_stats = {
                "axis_index": axis_index,
                "title": self._get_axis_title(ax),
                "data_types": data_types,
                "data_points": 0,
                "matrix_data": None,
                "x_type": x_type,
                "y_type": y_type,
                "curve_points": []
            }

            # Procesar gráficos de barras
            if "bar_plot" in data_types:
                total_value = 0
                category_values = []
                
                # Asegurarse de que el canvas esté dibujado para obtener las etiquetas
                if hasattr(ax.figure, "canvas"):
                    ax.figure.canvas.draw()
                
                for idx, patch in enumerate(ax.patches):
                    if hasattr(patch, "get_x") and hasattr(patch, "get_height"):
                        x = patch.get_x()
                        y = float(patch.get_height())  # Convertir a float nativo
                        # Usar etiquetas de categoría si están disponibles
                        x_val = x_labels[idx] if x_type == "category" and idx < len(x_labels) else float(x)
                        
                        curve_point = {
                            "x": [x_val],
                            "y": [y],
                            "label": getattr(patch, "get_label", lambda: None)()
                        }
                        axis_stats["curve_points"].append(curve_point)
                        
                        category_values.append({"category": x_val, "value": y})
                        total_value += y
                
                # Calcular y agregar estadísticas por categoría
                for cat_val in category_values:
                    statistics["per_curve"].append({
                        "axis_index": axis_index,
                        "category": cat_val["category"],
                        "value": float(cat_val["value"]),
                        "percentage": float(cat_val["value"] / total_value * 100) if total_value > 0 else 0.0
                    })
                
                # Actualizar el número de puntos de datos
                axis_stats["data_points"] = len(category_values)
            
            # Calcular estadísticas si los datos son numéricos
            if y_type == "numeric" and axis_stats["curve_points"]:
                y_data = [float(point) for cp in axis_stats["curve_points"] for point in cp["y"]]
                if y_data:
                    y_array = np.array(y_data)
                    axis_stats.update({
                        "mean": float(np.nanmean(y_array)),
                        "std": float(np.nanstd(y_array)),
                        "min": float(np.nanmin(y_array)),
                        "max": float(np.nanmax(y_array)),
                        "median": float(np.nanmedian(y_array)),
                        "outliers": [],
                        "local_var": 0.0,
                        "trend": None,
                        "skewness": float(self._calculate_skewness(y_array)),
                        "kurtosis": float(self._calculate_kurtosis(y_array))
                    })

            statistics["per_axis"].append(axis_stats)

        # Convertir todos los valores NumPy a tipos Python nativos
        return to_native_type(statistics)

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

        colors: List[Dict[str, Any]] = []
        try:
            axes = self._get_axes(figure)
            for ax in axes:
                # NO colors from images (heatmaps)
                # Only extract from lines, collections, patches
                # Colors from lines
                for line in ax.lines:
                    if hasattr(line, "_color"):
                        try:
                            color_hex = to_hex(line._color)
                            color_name = hex_to_name(color_hex)
                            if color_hex not in [c["hex"] for c in colors]:
                                colors.append({"hex": color_hex, "name": color_name})
                        except Exception:
                            continue
                # Colors from collections (scatter plots)
                for collection in ax.collections:
                    if hasattr(collection, "_facecolors"):
                        for color in collection._facecolors:
                            try:
                                hex_color = to_hex(color)
                                color_name = hex_to_name(hex_color)
                                if hex_color not in [c["hex"] for c in colors]:
                                    colors.append(
                                        {"hex": hex_color, "name": color_name}
                                    )
                            except Exception:
                                continue
                # Colors from patches (histograms, bar plots)
                for patch in ax.patches:
                    try:
                        if hasattr(patch, "get_facecolor"):
                            facecolor = patch.get_facecolor()
                            if facecolor is not None:
                                try:
                                    hex_color = to_hex(facecolor)
                                    color_name = hex_to_name(hex_color)
                                    if hex_color not in [c["hex"] for c in colors]:
                                        colors.append(
                                            {"hex": hex_color, "name": color_name}
                                        )
                                except Exception:
                                    continue
                    except Exception:
                        continue
        except Exception as e:
            logger.warning(f"Error extracting colors: {str(e)}")
        return colors

    def _get_markers(self, figure: Any) -> List[dict]:
        """Get the markers used in the figure, as readable codes and names."""
        markers: List[Dict[str, Any]] = []
        try:
            axes = self._get_axes(figure)
            for ax in axes:
                for line in ax.lines:
                    marker_code = (
                        line.get_marker() if hasattr(line, "get_marker") else None
                    )
                    if (
                        marker_code
                        and marker_code != "None"
                        and marker_code not in [m["code"] for m in markers]
                    ):
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
                "-": "solid",
                "--": "dashed",
                "-.": "dashdot",
                ":": "dotted",
                "None": "none",
                " ": "none",
                "": "none",
            }
            return style_names.get(str(style_code), str(style_code))

        styles: List[Dict[str, Any]] = []
        try:
            axes = self._get_axes(figure)

            for ax in axes:
                for line in ax.lines:
                    if hasattr(line, "_linestyle") and line._linestyle != "None":
                        style_code = line._linestyle
                        style_name = line_style_to_name(style_code)
                        if style_code not in [s["code"] for s in styles]:
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
                        "color": (
                            to_hex(line._color) if hasattr(line, "_color") else None
                        ),
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
                        "edgecolors": (
                            [to_hex(c) for c in collection.get_edgecolors()]
                            if hasattr(collection, "get_edgecolors")
                            else []
                        ),
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
                "axes_count": axes_count,
            }
        except Exception as e:
            logger.warning(f"Error getting figure info: {str(e)}")
            return {
                "figure_type": "unknown",
                "dimensions": [0, 0],
                "title": None,
                "axes_count": 0,
            }

    def _get_axis_info(self, figure: Any) -> Dict[str, Any]:
        """Get detailed information about axes, including titles and labels."""
        axis_info = {"axes": [], "figure_title": "", "total_axes": 0}

        try:
            axes = self._get_axes(figure)
            axis_info["total_axes"] = len(axes)

            # Get figure title
            if hasattr(figure, "_suptitle") and figure._suptitle:
                axis_info["figure_title"] = figure._suptitle.get_text()
            elif hasattr(figure, "get_suptitle"):
                axis_info["figure_title"] = figure.get_suptitle()

            for i, ax in enumerate(axes):
                ax_info = {
                    "index": i,
                    "title": "",
                    "x_label": "",
                    "y_label": "",
                    "x_lim": None,
                    "y_lim": None,
                    "has_data": False,
                    "plot_types": [],
                    "x_type": None,
                    "y_type": None,
                    "has_grid": False,
                    "has_legend": False,
                }

                # Extract axis title (subplot title)
                try:
                    if hasattr(ax, "get_title"):
                        title = ax.get_title()
                        if title and title.strip():
                            ax_info["title"] = title.strip()
                except Exception:
                    pass

                # Extract X and Y axis labels
                try:
                    if hasattr(ax, "get_xlabel"):
                        x_label = ax.get_xlabel()
                        if x_label and x_label.strip():
                            ax_info["x_label"] = x_label.strip()
                except Exception:
                    pass

                try:
                    if hasattr(ax, "get_ylabel"):
                        y_label = ax.get_ylabel()
                        if y_label and y_label.strip():
                            ax_info["y_label"] = y_label.strip()
                except Exception:
                    pass

                # Extract axis limits
                try:
                    if hasattr(ax, "get_xlim"):
                        x_lim = ax.get_xlim()
                        if x_lim and len(x_lim) == 2:
                            ax_info["x_lim"] = [float(x_lim[0]), float(x_lim[1])]
                except Exception:
                    pass

                try:
                    if hasattr(ax, "get_ylim"):
                        y_lim = ax.get_ylim()
                        if y_lim and len(y_lim) == 2:
                            ax_info["y_lim"] = [float(y_lim[0]), float(y_lim[1])]
                except Exception:
                    pass

                # Check grid and legend
                try:
                    ax_info["has_grid"] = ax.get_xgrid() or ax.get_ygrid()
                except Exception:
                    pass

                try:
                    ax_info["has_legend"] = ax.get_legend() is not None
                except Exception:
                    pass

                # Detect plot types and axis types
                plot_types: List[Dict[str, Any]] = []
                x_type: Optional[str] = None
                y_type: Optional[str] = None

                # Check lines (line plots)
                if hasattr(ax, "lines") and ax.lines:
                    plot_types.append({"type": "line"})
                    for line in ax.lines:
                        x = line.get_xdata()
                        y = line.get_ydata()
                        if x_type is None:
                            import numpy as np

                            if np.issubdtype(np.array(x).dtype, np.datetime64):
                                x_type = "DATE"
                            elif hasattr(x, "dtype") and str(x.dtype).startswith(
                                "period"
                            ):
                                x_type = "PERIOD"
                            elif all(isinstance(val, str) for val in x):
                                x_type = "CATEGORY"
                            else:
                                x_type = "NUMERIC"
                        if y_type is None:
                            import numpy as np

                            if np.issubdtype(np.array(y).dtype, np.datetime64):
                                y_type = "DATE"
                            elif all(isinstance(val, str) for val in y):
                                y_type = "CATEGORY"
                            else:
                                y_type = "NUMERIC"

                # Check collections (scatter plots)
                if hasattr(ax, "collections") and ax.collections:
                    plot_types.append({"type": "scatter"})
                    for collection in ax.collections:
                        if hasattr(collection, "get_offsets"):
                            offsets = collection.get_offsets()
                            if offsets is not None and len(offsets) > 0:
                                x = offsets[:, 0]
                                y = offsets[:, 1]
                                if x_type is None:
                                    import numpy as np

                                    if np.issubdtype(np.array(x).dtype, np.datetime64):
                                        x_type = "DATE"
                                    elif all(isinstance(val, str) for val in x):
                                        x_type = "CATEGORY"
                                    else:
                                        x_type = "NUMERIC"
                                if y_type is None:
                                    import numpy as np

                                    if np.issubdtype(np.array(y).dtype, np.datetime64):
                                        y_type = "DATE"
                                    elif all(isinstance(val, str) for val in y):
                                        y_type = "CATEGORY"
                                    else:
                                        y_type = "NUMERIC"

                # Check patches (bar plots, histograms)
                if hasattr(ax, "patches") and ax.patches:
                    # Determine if it's bar or histogram based on patch properties
                    is_bar = False
                    is_histogram = False

                    for patch in ax.patches:
                        if hasattr(patch, "get_x") and hasattr(patch, "get_height"):
                            # Check if patches are adjacent (bar plot) or overlapping (histogram)
                            if len(ax.patches) > 1:
                                width = patch.get_width()
                                # Simple heuristic: if patches are close together, it's likely a bar plot
                                if width > 0.1:  # Bar plots typically have wider bars
                                    is_bar = True
                                else:
                                    is_histogram = True
                            else:
                                is_bar = True
                            break

                    if is_bar:
                        plot_types.append({"type": "bar"})
                    elif is_histogram:
                        plot_types.append({"type": "histogram"})

                    # For bar plots, check if x-axis has categorical labels
                    if is_bar and hasattr(ax, "get_xticklabels"):
                        try:
                            xticklabels = ax.get_xticklabels()
                            if xticklabels and all(
                                isinstance(label.get_text(), str)
                                for label in xticklabels
                            ):
                                x_type = "CATEGORY"
                            else:
                                x_type = "NUMERIC"
                        except Exception:
                            x_type = "NUMERIC"

                    if y_type is None:
                        y_type = "NUMERIC"

                # Check images (heatmaps)
                if hasattr(ax, "images") and ax.images:
                    plot_types.append({"type": "heatmap"})
                    if x_type is None:
                        x_type = "NUMERIC"
                    if y_type is None:
                        y_type = "NUMERIC"

                # Set default types if not detected
                if x_type is None:
                    x_type = "NUMERIC"
                if y_type is None:
                    y_type = "NUMERIC"

                ax_info["plot_types"] = plot_types
                ax_info["x_type"] = x_type
                ax_info["y_type"] = y_type

                # Check if axis has data
                try:
                    has_data = False

                    # Check collections (scatter plots, etc.)
                    if hasattr(ax, "collections") and ax.collections:
                        has_data = True

                    # Check lines
                    if hasattr(ax, "lines") and ax.lines:
                        has_data = True

                    # Check patches (histograms, bar plots)
                    if hasattr(ax, "patches") and ax.patches:
                        has_data = True

                    # Check images (heatmaps)
                    if hasattr(ax, "images") and ax.images:
                        has_data = True

                    ax_info["has_data"] = has_data
                except Exception:
                    ax_info["has_data"] = False

                axis_info["axes"].append(ax_info)

        except Exception as e:
            logger.warning(f"Error getting axis info: {str(e)}")

        return axis_info
