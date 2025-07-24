"""
Matplotlib-specific analyzer for extracting information from matplotlib figures.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.axes as mpl_axes
import matplotlib.figure as mpl_figure
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.markers import MarkerStyle

from plot2llm.utils import serialize_axis_values

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
            axis_info = self._get_axis_info(figure)
            colors = self._get_colors(figure) if include_colors else []
            statistics = (
                self._get_statistics(figure)
                if include_statistics
                else {"per_curve": [], "per_axis": []}
            )

            # For bar/categorical plots, ensure xticklabels are available
            if hasattr(figure, "canvas"):
                try:
                    figure.canvas.draw()
                except Exception:
                    pass

            # Compose axes list for compatibility
            axes: List[Dict[str, Any]] = []
            real_axes = self._get_axes(figure)
            for idx, ax_info in enumerate(axis_info["axes"]):
                plot_types: List[Dict[str, Any]] = []
                curve_points: List[Dict[str, Any]] = []
                x_type: Optional[str] = None
                y_type: Optional[str] = None
                has_grid = ax_info.get("has_grid", False)
                has_legend = ax_info.get("has_legend", False)
                x_range = ax_info.get("x_lim")
                y_range = ax_info.get("y_lim")
                spine_visibility = None  # Placeholder, could be improved
                tick_density = None      # Placeholder, could be improved
                if idx < len(real_axes):
                    ax = real_axes[idx]
                    # Detectar tipo de eje y etiquetas para x/y
                    x_type_detected, x_labels = self._detect_axis_type_and_labels(ax, "x")
                    y_type_detected, y_labels = self._detect_axis_type_and_labels(ax, "y")
                    x_type = x_type_detected
                    y_type = y_type_detected
                    # Detectar plot types y curve_points
                    # Line plots
                    if hasattr(ax, "lines") and ax.lines:
                        plot_types.append({"type": "line"})
                        for line in ax.lines:
                            x = line.get_xdata()
                            y = line.get_ydata()
                            x_serial = serialize_axis_values(x)
                            y_serial = serialize_axis_values(y)
                            curve_points.append({"x": x_serial, "y": y_serial, "label": line.get_label()})
                    # Scatter plots
                    if hasattr(ax, "collections") and ax.collections:
                        plot_types.append({"type": "scatter"})
                        for collection in ax.collections:
                            if hasattr(collection, "get_offsets"):
                                offsets = collection.get_offsets()
                                if offsets is not None and len(offsets) > 0:
                                    x = offsets[:, 0]
                                    y = offsets[:, 1]
                                    x_serial = serialize_axis_values(x)
                                    y_serial = serialize_axis_values(y)
                                    curve_points.append({"x": x_serial, "y": y_serial, "label": getattr(collection, "get_label", lambda: None)()})
                    # Bar/histogram
                    if hasattr(ax, "patches") and ax.patches:
                        for patch_idx, patch in enumerate(ax.patches):
                            if hasattr(patch, "get_x") and hasattr(patch, "get_height"):
                                x = patch.get_x()
                                y = patch.get_height()
                                x_val = x_labels[patch_idx] if x_type == "category" and patch_idx < len(x_labels) else x
                                x_serial = [x_val] if isinstance(x_val, str) else serialize_axis_values([x_val])
                                y_serial = serialize_axis_values([y])
                                curve_points.append({"x": x_serial, "y": y_serial, "label": getattr(patch, "get_label", lambda: None)()})
                        plot_types.append({"type": "bar"})
                
                # --- Analysis ---
                x_data_list = []
                y_data_list = []
                for p in curve_points:
                    x_data_list.extend(p.get("x", []))
                    y_data_list.extend(p.get("y", []))

                x_data = np.array(x_data_list)
                y_data = np.array(y_data_list)

                # --- Pattern and Shape Analysis ---
                pattern_results = self._analyze_patterns(x_data, y_data) if x_data.size > 0 and y_data.size > 0 else {}
                shape_results = self._analyze_shape_characteristics(y_data) if y_data.size > 0 else {}
                
                # --- Domain Context Intelligence ---
                domain = self._infer_domain(
                    ax_info.get("title", ""),
                    ax_info.get("x_label", ""),
                    ax_info.get("y_label", ""),
                    pattern_results.get("pattern_type") if pattern_results else None
                )
                purpose = self._infer_purpose(
                    pattern_results,
                    len(curve_points),
                    x_type
                )
                complexity = self._assess_complexity_level(
                    pattern_results,
                    2 # Assuming 2 variables for now
                )
                domain_context = {
                    "likely_domain": domain,
                    "purpose_inference": purpose,
                    "complexity_level": complexity,
                }

                # --- Statistical Insights ---
                trend = self._detect_trend(y_data, x_data)
                distribution = self._analyze_distribution(y_data)
                outliers = self._detect_outliers(y_data)
                correlations = self._calculate_correlations(x_data, y_data)
                key_statistics = {
                    "mean": float(np.nanmean(y_data)) if y_data.size > 0 else 0,
                    "median": float(np.nanmedian(y_data)) if y_data.size > 0 else 0,
                    "std": float(np.nanstd(y_data)) if y_data.size > 0 else 0,
                    "range": (float(np.nanmin(y_data)), float(np.nanmax(y_data))) if y_data.size > 0 else (0, 0),
                }

                # --- Build Enriched Axis ---
                current_axis = {
                    "title": ax_info.get("title", ""),
                    "xlabel": ax_info.get("x_label", ""),
                    "ylabel": ax_info.get("y_label", ""),
                    "plot_types": plot_types,
                    "curve_points": curve_points,
                    "x_type": x_type,
                    "y_type": y_type,
                    "has_grid": has_grid,
                    "has_legend": has_legend,
                    "x_range": x_range,
                    "y_range": y_range,
                    "spine_visibility": spine_visibility,
                    "tick_density": tick_density,
                    "pattern": pattern_results,
                    "shape": shape_results,
                    "domain_context": {
                        "likely_domain": domain,
                        "purpose": purpose,
                        "complexity_level": complexity,
                        "mathematical_properties": {
                            "function_type": pattern_results.get("pattern_type"),
                            "is_analytical": pattern_results.get("confidence_score", 0) > 0.9,
                            "theoretical_vs_measured": "theoretical" if pattern_results.get("confidence_score", 0) > 0.95 else "measured"
                        }
                    },
                    "stats": {
                        "trend": trend,
                        "distribution": distribution,
                        "outliers": outliers,
                        "key_statistics": key_statistics
                    },
                    "x_semantics": x_type,  # Agregar semantics para mejorar el when
                    "y_semantics": y_type   # Agregar semantics para mejorar el why
                }

                # --- Data Summary ---
                x_vals = [pt for cp in curve_points for pt in cp.get("x", [])]
                y_vals = [pt for cp in curve_points for pt in cp.get("y", [])]
                total_data_points = sum(len(cp.get("y", [])) for cp in curve_points)

                data_summary = {
                    "total_data_points": total_data_points,
                    "data_ranges": {
                        "x": {
                            "min": float(np.min(x_vals)) if x_vals else None,
                            "max": float(np.max(x_vals)) if x_vals else None,
                            "type": x_type
                        },
                        "y": {
                            "min": float(np.min(y_vals)) if y_vals else None,
                            "max": float(np.max(y_vals)) if y_vals else None,
                            "type": y_type
                        }
                    },
                    "missing_values": {
                        "x": sum(1 for x in x_vals if x is None or (isinstance(x, float) and np.isnan(x))),
                        "y": sum(1 for y in y_vals if y is None or (isinstance(y, float) and np.isnan(y)))
                    },
                    "x_type": x_type,
                    "y_type": y_type
                }

                current_axis["data_summary"] = data_summary
                axes.append(current_axis)

            # This part is tricky as there's one analysis per figure, but we are in a loop per axis.
            # For now, let's just overwrite with the last axis' analysis.
            if 'analysis_results' in locals():
                figure_analysis = analysis_results

            # Compose output
            title = figure_info.get("title", "")
            if not title and axes and axes[0].get("title"):
                title = axes[0]["title"]
            result = {
                "figure_type": "matplotlib",
                "title": title,
                "axes": axes,
                "basic_info": figure_info,
                "axes_info": axis_info["axes"],
                "data_info": {
                    "plot_types": [pt for ax in axes for pt in ax.get("plot_types", [])],
                    "statistics": statistics,
                },
                "visual_info": {"colors": colors},
                "statistics": statistics,
            }
            if 'figure_analysis' in locals():
                result.update(figure_analysis)

            if detail_level == "high":
                result["detailed_info"] = self._extract_detailed_info(figure)
            return result
        except Exception as e:
            logger.error(f"Error analyzing matplotlib figure: {str(e)}")
            return {
                "figure_type": "matplotlib",
                "title": None,
                "axes": [],
                "basic_info": {},
                "axes_info": [],
                "data_info": {"plot_types": [], "statistics": {}},
                "visual_info": {"colors": []},
                "statistics": {"per_curve": [], "per_axis": []},
                "error": str(e),
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
        """Get statistical information about the data in the matplotlib figure, per curve and per axis."""
        stats = {"per_curve": [], "per_axis": []}
        try:
            axes = self._get_axes(figure)
            for i, ax in enumerate(axes):
                axis_stats = {
                    "axis_index": i,
                    "title": (
                        ax.get_title()
                        if hasattr(ax, "get_title") and ax.get_title()
                        else f"Subplot {i+1}"
                    ),
                    "data_types": [],
                    "data_points": 0,
                    "matrix_data": None,
                }

                # Detectar tipo de eje y etiquetas
                x_type, x_labels = self._detect_axis_type_and_labels(ax, "x")
                y_type, y_labels = self._detect_axis_type_and_labels(ax, "y")

                # Extraer puntos de la curva para este eje
                curve_points: List[Dict[str, Any]] = []

                # Line plots
                if hasattr(ax, "lines") and ax.lines:
                    for line in ax.lines:
                        x = line.get_xdata()
                        y = line.get_ydata()
                        x_serial = (
                            x_labels
                            if x_type == "category"
                            else serialize_axis_values(x)
                        )
                        y_serial = (
                            y_labels
                            if y_type == "category"
                            else serialize_axis_values(y)
                        )
                        curve_points.append(
                            {"x": x_serial, "y": y_serial, "label": line.get_label()}
                        )

                # Scatter plots
                if hasattr(ax, "collections") and ax.collections:
                    for collection in ax.collections:
                        if hasattr(collection, "get_offsets"):
                            offsets = collection.get_offsets()
                            if offsets is not None and len(offsets) > 0:
                                x = offsets[:, 0]
                                y = offsets[:, 1]
                                x_serial = (
                                    x_labels
                                    if x_type == "category"
                                    else serialize_axis_values(x)
                                )
                                y_serial = (
                                    y_labels
                                    if y_type == "category"
                                    else serialize_axis_values(y)
                                )
                                curve_points.append(
                                    {
                                        "x": x_serial,
                                        "y": y_serial,
                                        "label": getattr(
                                            collection, "get_label", lambda: None
                                        )(),
                                    }
                                )

                # Bar/histogram
                if hasattr(ax, "patches") and ax.patches:
                    for idx, patch in enumerate(ax.patches):
                        if hasattr(patch, "get_x") and hasattr(patch, "get_height"):
                            x = patch.get_x()
                            y = patch.get_height()
                            # Usar etiquetas de categoría si están disponibles
                            x_val = (
                                x_labels[idx]
                                if x_type == "category" and idx < len(x_labels)
                                else x
                            )
                            x_serial = (
                                [x_val]
                                if isinstance(x_val, str)
                                else serialize_axis_values([x_val])
                            )
                            y_serial = serialize_axis_values([y])
                            curve_points.append(
                                {
                                    "x": x_serial,
                                    "y": y_serial,
                                    "label": getattr(
                                        patch, "get_label", lambda: None
                                    )(),
                                }
                            )

                # Determinar si se pueden calcular estadísticas
                can_calc_stats = (
                    y_type == "numeric"
                )  # Solo calcular estadísticas si Y es numérico

                # Extraer datos Y para estadísticas
                y_data = []
                for pt in curve_points:
                    if isinstance(pt["y"], (list, tuple)):
                        y_data.extend(pt["y"])
                    else:
                        y_data.append(pt["y"])

                y_data = np.array(y_data)
                if (
                    can_calc_stats
                    and len(y_data) > 0
                    and np.issubdtype(y_data.dtype, np.number)
                ):
                    axis_stats.update(
                        {
                            "mean": float(np.nanmean(y_data)),
                            "std": float(np.nanstd(y_data)),
                            "min": float(np.nanmin(y_data)),
                            "max": float(np.nanmax(y_data)),
                            "median": float(np.nanmedian(y_data)),
                            "outliers": [],
                            "local_var": float(
                                np.nanvar(y_data[: max(1, len(y_data) // 10)])
                            ),
                            "trend": None,
                            "skewness": self._calculate_skewness(y_data),
                            "kurtosis": self._calculate_kurtosis(y_data),
                        }
                    )
                else:
                    axis_stats.update(
                        {
                            "mean": None,
                            "std": None,
                            "min": None,
                            "max": None,
                            "median": None,
                            "outliers": [],
                            "local_var": None,
                            "trend": None,
                            "skewness": None,
                            "kurtosis": None,
                        }
                    )

                # Actualizar información del eje
                axis_stats["x_type"] = x_type
                axis_stats["y_type"] = y_type
                axis_stats["curve_points"] = curve_points
                axis_stats["data_types"].extend(
                    [
                        (
                            "line_plot"
                            if hasattr(ax, "lines") and ax.lines
                            else "scatter_plot"
                        )
                    ]
                )
                axis_stats["data_points"] = sum(
                    len(pt["y"]) if isinstance(pt["y"], (list, tuple)) else 1
                    for pt in curve_points
                )

                stats["per_axis"].append(axis_stats)

            return stats

        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                f"Error calculating matplotlib statistics: {str(e)}"
            )
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
