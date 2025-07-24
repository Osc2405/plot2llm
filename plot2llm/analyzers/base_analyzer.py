"""
Base analyzer class that defines the interface for all figure analyzers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import shapiro

from plot2llm.domain_knowledge import DOMAIN_KEYWORDS

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

    # --- Pattern Analysis Methods ---
    def _detect_linear_pattern(self, x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detects a linear pattern (y = mx + b) in the data.

        Args:
            x: X-coordinates of the data points.
            y: Y-coordinates of the data points.

        Returns:
            A tuple (m, b) representing the slope and intercept of the fitted line,
            or None if a pattern cannot be determined.
        """
        if len(x) < 2 or len(y) < 2 or len(x) != len(y):
            return None
        try:
            # Fit a first-degree polynomial (a line)
            coeffs = np.polyfit(x, y, 1)
            if len(coeffs) == 2:
                return float(coeffs[0]), float(coeffs[1])
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Could not fit linear pattern: {e}")
            return None
        return None

    def _detect_polynomial_pattern(self, x: np.ndarray, y: np.ndarray, max_degree: int = 4) -> Optional[np.ndarray]:
        """
        Detects a polynomial pattern (y = a*x^n + b*x^(n-1) + ... + z) in the data.

        Args:
            x: X-coordinates of the data points.
            y: Y-coordinates of the data points.
            max_degree: The maximum polynomial degree to test.

        Returns:
            The coefficients of the best-fit polynomial, or None.
        """
        if len(x) < max_degree + 1 or len(y) < max_degree + 1 or len(x) != len(y):
            return None
        try:
            best_coeffs = None
            best_error = np.inf
            for degree in range(2, max_degree + 1):
                coeffs = np.polyfit(x, y, degree)
                model = np.poly1d(coeffs)
                error = np.sum((model(x) - y) ** 2)
                if error < best_error:
                    best_error = error
                    best_coeffs = coeffs
            return best_coeffs
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Could not fit polynomial pattern: {e}")
            return None
        return None

    def _detect_exponential_pattern(self, x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detects an exponential pattern (y = a * exp(b*x)).
        """
        if len(x) < 2 or len(y) < 2 or len(x) != len(y) or np.any(y <= 0):
            return None
        try:
            # y = a * exp(b*x) => log(y) = log(a) + b*x
            log_y = np.log(y)
            coeffs = np.polyfit(x, log_y, 1)
            b = coeffs[0]
            log_a = coeffs[1]
            a = np.exp(log_a)
            return a, b
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Could not fit exponential pattern: {e}")
            return None
        return None

    def _detect_logarithmic_pattern(self, x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detects a logarithmic pattern (y = a * log(x) + b).
        """
        if len(x) < 2 or len(y) < 2 or len(x) != len(y) or np.any(x <= 0):
            return None
        try:
            # y = a * log(x) + b is a linear relationship between y and log(x)
            log_x = np.log(x)
            coeffs = np.polyfit(log_x, y, 1)
            return float(coeffs[0]), float(coeffs[1])
        except (np.linalg.LinAlgError, ValueError) as e:
            logger.warning(f"Could not fit logarithmic pattern: {e}")
            return None
        return None

    def _detect_sinusoidal_pattern(self, x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """
        Detects a sinusoidal pattern (y = a*sin(b*x + c) + d).
        """
        if len(x) < 5 or len(y) < 5 or len(x) != len(y):
            return None
        try:
            from scipy.optimize import curve_fit

            def sin_func(x, a, b, c, d):
                return a * np.sin(b * x + c) + d

            # Provide initial guesses for the parameters
            guess_freq = (2 * np.pi) / (x[np.argmax(np.abs(np.fft.fftfreq(len(x))))] * (x[1] - x[0])) if len(x) > 1 else 1
            guess_amp = np.std(y) * 2**0.5
            guess_offset = np.mean(y)
            guess = [guess_amp, guess_freq, 0, guess_offset]

            params, _ = curve_fit(sin_func, x, y, p0=guess, maxfev=10000)
            return tuple(params)
        except ImportError:
            logger.warning("Scipy not installed, skipping sinusoidal pattern detection.")
            return None
        except (RuntimeError, ValueError) as e:
            logger.warning(f"Could not fit sinusoidal pattern: {e}")
            return None
        return None

    def _calculate_confidence_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the confidence score (R-squared) for a model's predictions.
        """
        if len(y_true) == 0 or len(y_pred) == 0 or len(y_true) != len(y_pred):
            return 0.0
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            # Handle the case where all y values are the same
            return 1.0 if ss_res == 0 else 0.0
            
        r2 = 1 - (ss_res / ss_tot)
        return max(0.0, r2)  # R-squared can be negative, clip to 0

    def _calculate_aic(self, n: int, sse: float, k: int) -> float:
        """
        Calculates the Akaike Information Criterion (AIC) for a model.
        """
        if n <= 0 or sse < 0 or k <= 0:
            return np.inf
        
        aic = n * np.log(sse / n) + 2 * k
        return aic
        
    def _analyze_patterns(self, x: np.ndarray, y: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Analyzes the data for various patterns and returns the best fit based on AIC.
        """
        if len(x) < 2 or len(y) < 2:
            return None

        patterns = []

        # Linear
        linear_coeffs = self._detect_linear_pattern(x, y)
        if linear_coeffs:
            m, b = linear_coeffs
            y_pred = m * x + b
            score = self._calculate_confidence_score(y, y_pred)
            sse = np.sum((y - y_pred) ** 2)
            aic = self._calculate_aic(len(y), sse, 2)
            patterns.append({"type": "linear", "coeffs": (m, b), "score": score, "aic": aic, "equation": f"y = {m:.2f}x + {b:.2f}"})

        # Polynomial
        for degree in range(2, 5):
            poly_coeffs = np.polyfit(x, y, degree)
            model = np.poly1d(poly_coeffs)
            y_pred = model(x)
            score = self._calculate_confidence_score(y, y_pred)
            sse = np.sum((y - y_pred) ** 2)
            aic = self._calculate_aic(len(y), sse, degree + 1)
            patterns.append({"type": f"polynomial (deg {degree})", "coeffs": poly_coeffs, "score": score, "aic": aic, "equation": str(model)})

        # Exponential
        exp_coeffs = self._detect_exponential_pattern(x, y)
        if exp_coeffs:
            a, b = exp_coeffs
            y_pred = a * np.exp(b * x)
            score = self._calculate_confidence_score(y, y_pred)
            sse = np.sum((y - y_pred) ** 2)
            aic = self._calculate_aic(len(y), sse, 2)
            patterns.append({"type": "exponential", "coeffs": (a, b), "score": score, "aic": aic, "equation": f"y = {a:.2f} * exp({b:.2f}x)"})

        # Logarithmic
        log_coeffs = self._detect_logarithmic_pattern(x, y)
        if log_coeffs:
            a, b = log_coeffs
            y_pred = a * np.log(x) + b
            score = self._calculate_confidence_score(y, y_pred)
            sse = np.sum((y - y_pred) ** 2)
            aic = self._calculate_aic(len(y), sse, 2)
            patterns.append({"type": "logarithmic", "coeffs": (a, b), "score": score, "aic": aic, "equation": f"y = {a:.2f} * log(x) + {b:.2f}"})
            
        # Sinusoidal
        sin_coeffs = self._detect_sinusoidal_pattern(x, y)
        if sin_coeffs:
            a, b, c, d = sin_coeffs
            y_pred = a * np.sin(b * x + c) + d
            score = self._calculate_confidence_score(y, y_pred)
            sse = np.sum((y - y_pred) ** 2)
            aic = self._calculate_aic(len(y), sse, 4)
            patterns.append({"type": "sinusoidal", "coeffs": (a, b, c, d), "score": score, "aic": aic, "equation": f"y = {a:.2f}sin({b:.2f}x + {c:.2f}) + {d:.2f}"})

        if not patterns:
            return None

        # Return the best pattern based on AIC
        best_pattern = min(patterns, key=lambda p: p["aic"])
        return {
            "pattern_type": best_pattern["type"],
            "equation_estimate": best_pattern["equation"],
            "coefficients": best_pattern["coeffs"],
            "confidence_score": best_pattern["score"]
        }

    def _analyze_shape_characteristics(self, y: np.ndarray) -> Dict[str, str]:
        """
        Analyzes the shape characteristics of the data.
        """
        if len(y) < 3:
            return {}

        diffs = np.diff(y)
        
        # Monotonicity
        if np.all(diffs >= 0):
            monotonicity = "increasing"
        elif np.all(diffs <= 0):
            monotonicity = "decreasing"
        else:
            monotonicity = "mixed"

        # Smoothness
        second_deriv = np.diff(y, 2)
        if np.all(np.abs(second_deriv) < 1e-5):
            smoothness = "smooth"
        elif np.mean(np.abs(second_deriv)) < 0.1 * np.mean(np.abs(diffs)):
            smoothness = "piecewise"
        else:
            smoothness = "discrete"

        # Symmetry
        mean = np.mean(y)
        skewness = np.mean(((y - mean) / np.std(y)) ** 3) if np.std(y) > 0 else 0
        if abs(skewness) < 0.5:
            symmetry = "symmetric"
        else:
            symmetry = "asymmetric"

        # Continuity
        if np.max(np.abs(diffs)) > 5 * np.median(np.abs(diffs)):
            continuity = "discontinuous"
        else:
            continuity = "continuous"

        return {
            "monotonicity": monotonicity,
            "smoothness": smoothness,
            "symmetry": symmetry,
            "continuity": continuity
        }

    def _infer_domain(self, title: str, xlabel: str, ylabel: str, pattern_type: Optional[str]) -> str:
        """
        Infers the likely domain of the plot based on keywords and patterns.
        """
        text_corpus = (title + " " + xlabel + " " + ylabel).lower()
        
        domain_scores = {domain: 0 for domain in DOMAIN_KEYWORDS}

        # Score based on keywords
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_corpus:
                    domain_scores[domain] += 2 # Higher weight for keywords
        
        # Boost score based on pattern
        if pattern_type in ["linear", "polynomial", "exponential", "logarithmic", "sinusoidal"]:
            domain_scores["mathematics"] += 1 # Lower weight for pattern

        # Determine the best domain
        if not any(domain_scores.values()):
            return "general"
            
        best_domain = max(domain_scores, key=domain_scores.get)
        return best_domain

    def _infer_purpose(self, pattern_info: Optional[Dict[str, Any]], num_series: int, x_type: str) -> str:
        """
        Infers the likely purpose of the plot.
        """
        if pattern_info and pattern_info.get("confidence_score", 0) > 0.99:
            return "Educational"
        if num_series > 1:
            return "Comparison"
        if x_type == "date" or x_type == "period":
            return "Monitoring"
        
        return "Analysis"

    def _assess_complexity_level(self, pattern_info: Optional[Dict[str, Any]], num_variables: int) -> str:
        """
        Assesses the complexity level of the plot.
        """
        if pattern_info:
            pattern_type = pattern_info.get("pattern_type", "")
            if "polynomial" in pattern_type and "deg 4" in pattern_type:
                return "Advanced"
            if "sinusoidal" in pattern_type:
                return "Advanced"
            if "polynomial" in pattern_type:
                return "Intermediate"
        
        if num_variables > 2:
            return "Intermediate"

        return "Basic"

    # --- Statistical Insights Methods ---
    def _detect_trend(self, y: np.ndarray, x: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Detects the overall trend in a series of data.
        """
        if len(y) < 3:
            return None
        
        if x is None:
            x = np.arange(len(y))
        
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]

        if abs(slope) < 1e-1:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def _analyze_distribution(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes the distribution of a dataset.
        """
        if len(data) < 5:
            return {}
            
        skewness = np.mean(((data - np.mean(data)) / np.std(data)) ** 3) if np.std(data) > 0 else 0
        kurtosis = np.mean(((data - np.mean(data)) / np.std(data)) ** 4) - 3 if np.std(data) > 0 else 0
        
        # Normality test (Shapiro-Wilk)
        normality_test = None
        if len(data) >= 3:
            try:
                stat, p_value = shapiro(data)
                normality_test = {
                    "test": "shapiro-wilk",
                    "statistic": stat,
                    "p_value": p_value
                }
            except ImportError:
                logger.warning("Scipy is not installed, skipping normality test.")
            except Exception as e:
                logger.warning(f"Could not perform Shapiro-Wilk test: {e}")

        return {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "normality_test": normality_test
        }

    def _detect_outliers(self, data: np.ndarray) -> List[float]:
        """
        Detects outliers in a dataset using the IQR method.
        """
        if len(data) < 5:
            return []
            
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        outliers = [d for d in data if d < lower_bound or d > upper_bound]
        return outliers

    def _calculate_correlations(self, x: np.ndarray, y: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Calculates the Pearson correlation between two variables.
        """
        if len(x) < 2 or len(y) < 2 or len(x) != len(y):
            return None

        try:
            pearson_corr = np.corrcoef(x, y)[0, 1]
            return {"pearson": pearson_corr}
        except ValueError:
            return None

    def _detect_numeric_type(self, values):
        """Detect if values are numeric, even if they are strings representing numbers."""
        if not values:
            return None
        try:
            # Try to convert all to float
            floats = [float(v) for v in values]
            return "numeric"
        except Exception:
            # If all are strings but not numbers, treat as category
            if all(isinstance(v, str) for v in values):
                return "category"
            return None

    def _detect_temporal_type(self, values):
        """Detect if values are dates, timestamps, or temporal sequences."""
        import re, datetime
        if not values:
            return None
        # Check for datetime strings
        date_patterns = [
            r"\d{4}-\d{2}-\d{2}",  # YYYY-MM-DD
            r"\d{2}/\d{2}/\d{4}",  # DD/MM/YYYY
            r"\d{4}/\d{2}/\d{2}",  # YYYY/MM/DD
        ]
        for v in values:
            if isinstance(v, str):
                for pat in date_patterns:
                    if re.match(pat, v):
                        return "date"
            if isinstance(v, (datetime.date, datetime.datetime)):
                return "date"
        # Check for timestamps (large ints/floats)
        if all(isinstance(v, (int, float)) and v > 1e9 for v in values):
            return "timestamp"
        # Check for monotonic sequence (temporal index)
        if all(isinstance(v, (int, float)) for v in values):
            diffs = [values[i+1] - values[i] for i in range(len(values)-1)]
            if all(d > 0 for d in diffs):
                return "temporal_sequence"
        return None

    def _validate_type_consistency(self, axes):
        """Check that x_type and y_type are consistent across all axes."""
        x_types = set(ax.get("x_type") for ax in axes if ax.get("x_type"))
        y_types = set(ax.get("y_type") for ax in axes if ax.get("y_type"))
        return {
            "x_type_consistent": len(x_types) <= 1,
            "y_type_consistent": len(y_types) <= 1,
            "x_types": list(x_types),
            "y_types": list(y_types),
        }

    def _infer_axis_semantics(self, x_values, y_values):
        """Infer axis semantics for X and Y."""
        x_sem = None
        y_sem = None
        # X axis
        if self._detect_temporal_type(x_values) in ["date", "timestamp", "temporal_sequence"]:
            x_sem = "temporal"
        elif self._detect_numeric_type(x_values) == "numeric":
            x_sem = "continuous"
        else:
            x_sem = "category"
        # Y axis
        if self._detect_numeric_type(y_values) == "numeric":
            if all(isinstance(v, int) and v >= 0 for v in y_values):
                y_sem = "count"
            elif all(isinstance(v, float) and 0 <= v <= 1 for v in y_values):
                y_sem = "proportion"
            else:
                y_sem = "metric"
        else:
            y_sem = "category"
        return {"x_semantics": x_sem, "y_semantics": y_sem}
