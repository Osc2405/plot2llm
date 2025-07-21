"""
Base analyzer class that defines the interface for all figure analyzers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BaseAnalyzer(ABC):
    """
    Abstract base class for all figure analyzers.

    This class defines the interface that all specific analyzers must implement.
    """

    def __init__(self):
        """Initialize the base analyzer."""
        self.supported_types = []
        logger.debug(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def analyze(
        self,
        figure: Any,
        detail_level: str = "medium",
        include_data: bool = True,
        include_colors: bool = True,
        include_statistics: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze a figure and extract relevant information.

        Args:
            figure: The figure object to analyze
            detail_level: Level of detail ("low", "medium", "high")
            include_data: Whether to include data analysis
            include_colors: Whether to include color analysis
            include_statistics: Whether to include statistical analysis

        Returns:
            Dictionary containing the analysis results
        """
        pass

    def extract_basic_info(self, figure: Any) -> Dict[str, Any]:
        """
        Extract basic information from a figure.

        Args:
            figure: The figure object

        Returns:
            Dictionary with basic figure information
        """
        return {
            "figure_type": self._get_figure_type(figure),
            "dimensions": self._get_dimensions(figure),
            "title": self._get_title(figure),
            "axes_count": self._get_axes_count(figure),
        }

    def extract_axes_info(self, figure: Any) -> List[Dict[str, Any]]:
        """
        Extract information about all axes in the figure.

        Args:
            figure: The figure object

        Returns:
            List of dictionaries with axes information
        """
        axes_info = []
        try:
            axes = self._get_axes(figure)
            for i, ax in enumerate(axes):
                ax_info = {
                    "index": i,
                    "title": self._get_axis_title(ax),
                    "type": self._get_axis_type(ax),
                    "x_label": self._get_x_label(ax),
                    "y_label": self._get_y_label(ax),
                    "x_range": self._get_x_range(ax),
                    "y_range": self._get_y_range(ax),
                    "has_grid": self._has_grid(ax),
                    "has_legend": self._has_legend(ax),
                }
                axes_info.append(ax_info)
        except Exception as e:
            logger.warning(f"Error extracting axes info: {str(e)}")

        return axes_info

    def extract_data_info(self, figure: Any) -> Dict[str, Any]:
        """
        Extract data-related information from the figure.

        Args:
            figure: The figure object

        Returns:
            Dictionary with data information
        """
        try:
            return {
                "data_points": self._get_data_points(figure),
                "data_types": self._get_data_types(figure),
                "statistics": (
                    self._get_statistics(figure) if self.include_statistics else {}
                ),
            }
        except Exception as e:
            logger.warning(f"Error extracting data info: {str(e)}")
            return {}

    def extract_visual_info(self, figure: Any) -> Dict[str, Any]:
        """
        Extract visual information from the figure.

        Args:
            figure: The figure object

        Returns:
            Dictionary with visual information. Colors and markers are now lists of dicts with readable info.
        """
        try:
            return {
                "colors": (
                    self._get_colors(figure) if self.include_colors else []
                ),  # List[dict]
                "markers": self._get_markers(figure),  # List[dict]
                "line_styles": self._get_line_styles(figure),
                "background_color": self._get_background_color(figure),
            }
        except Exception as e:
            logger.warning(f"Error extracting visual info: {str(e)}")
            return {}

    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def _get_figure_type(self, figure: Any) -> str:
        """Get the type of the figure."""
        pass

    @abstractmethod
    def _get_dimensions(self, figure: Any) -> Tuple[int, int]:
        """Get the dimensions of the figure."""
        pass

    @abstractmethod
    def _get_title(self, figure: Any) -> Optional[str]:
        """Get the title of the figure."""
        pass

    @abstractmethod
    def _get_axes(self, figure: Any) -> List[Any]:
        """Get all axes in the figure."""
        pass

    @abstractmethod
    def _get_axes_count(self, figure: Any) -> int:
        """Get the number of axes in the figure."""
        pass

    # Optional methods with default implementations
    def _get_axis_type(self, ax: Any) -> str:
        """Get the type of an axis."""
        return "unknown"

    def _get_x_label(self, ax: Any) -> Optional[str]:
        """Get the x-axis label."""
        return None

    def _get_y_label(self, ax: Any) -> Optional[str]:
        """Get the y-axis label."""
        return None

    def _get_x_range(self, ax: Any) -> Optional[Tuple[float, float]]:
        """Get the x-axis range."""
        return None

    def _get_y_range(self, ax: Any) -> Optional[Tuple[float, float]]:
        """Get the y-axis range."""
        return None

    def _has_grid(self, ax: Any) -> bool:
        """Check if the axis has a grid."""
        return False

    def _has_legend(self, ax: Any) -> bool:
        """Check if the axis has a legend."""
        return False

    def _get_data_points(self, figure: Any) -> int:
        """Get the number of data points."""
        return 0

    def _get_data_types(self, figure: Any) -> List[str]:
        """Get the types of data in the figure."""
        return []

    def _get_statistics(self, figure: Any) -> dict:
        """Get statistical information about the data. Returns a dict with 'global' and 'per_curve'."""
        return {}

    def _get_colors(self, figure: Any) -> List[dict]:
        """Get the colors used in the figure, with hex and common name if possible."""
        return []

    def _get_markers(self, figure: Any) -> List[dict]:
        """Get the markers used in the figure, with codes and names."""
        return []

    def _get_line_styles(self, figure: Any) -> List[dict]:
        """Get the line styles used in the figure, with codes and names."""
        return []

    def _get_background_color(self, figure: Any) -> Optional[dict]:
        """Get the background color of the figure, with hex and common name if possible."""
        return None

    def _get_axis_title(self, ax: Any) -> Optional[str]:
        """Get the title of an individual axis."""
        return None
