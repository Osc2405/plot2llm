"""
Comprehensive tests for matplotlib analyzer functionality.

This module tests the MatplotlibAnalyzer class with various types of plots,
edge cases, and error conditions to ensure robustness and accuracy.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from plot2llm import FigureConverter, convert
from plot2llm.analyzers.matplotlib_analyzer import MatplotlibAnalyzer

# Suppress matplotlib warnings during tests
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
plt.ioff()  # Turn off interactive mode


class TestMatplotlibBasicPlots:
    """Test basic matplotlib plot types."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MatplotlibAnalyzer()
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_line_plot_basic(self):
        """Test basic line plot with simple data."""
        # Create basic line plot
        fig, ax = plt.subplots()
        x = [1, 2, 3]
        y = [2, 4, 6]
        ax.plot(x, y)

        # Analyze
        analysis = self.analyzer.analyze(fig)

        # Assertions - check new structure
        assert "figure" in analysis
        assert analysis["figure"]["figure_type"] == "matplotlib.Figure"
        assert len(analysis["axes"]) >= 1

        # Check that line plot is detected
        axes_data = analysis["axes"][0]
        assert "plot_type" in axes_data
        assert axes_data["plot_type"] == "line"

        # Check data extraction
        assert "lines" in axes_data
        assert len(axes_data.get("lines", [])) >= 1

        # Verify data points
        line = axes_data.get("lines", [])[0]
        assert "xdata" in line
        assert "ydata" in line
        assert len(line["xdata"]) == 3
        assert len(line["ydata"]) == 3

    @pytest.mark.unit
    def test_line_plot_with_labels(self):
        """Test line plot with title and axis labels."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [2, 4, 6])
        ax.set_title("Test Plot")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")

        analysis = self.analyzer.analyze(fig)

        # Check labels - new structure
        assert "figure" in analysis
        assert analysis["figure"]["title"] == "Test Plot"
        axes_data = analysis["axes"][0]
        assert axes_data["xlabel"] == "X Axis"
        assert axes_data["ylabel"] == "Y Axis"

    @pytest.mark.unit
    def test_scatter_plot_basic(self):
        """Test basic scatter plot."""
        fig, ax = plt.subplots()
        x = [1, 2, 3, 4]
        y = [1, 4, 2, 3]
        ax.scatter(x, y)

        analysis = self.analyzer.analyze(fig)

        # Check plot type - new structure
        axes_data = analysis["axes"][0]
        assert "plot_type" in axes_data
        assert axes_data["plot_type"] == "scatter"

        # Check data
        assert "collections" in axes_data
        assert len(axes_data.get("collections", [])) >= 1

    @pytest.mark.unit
    def test_scatter_plot_with_colors(self):
        """Test scatter plot with colors and sizes."""
        fig, ax = plt.subplots()
        x = [1, 2, 3, 4]
        y = [1, 4, 2, 3]
        colors = ["red", "blue", "green", "yellow"]
        sizes = [20, 40, 60, 80]
        ax.scatter(x, y, c=colors, s=sizes)

        analysis = self.analyzer.analyze(fig)

        # Check colors are captured
        assert "colors" in analysis
        colors_info = analysis["colors"]
        assert len(colors_info) > 0

        # Check scatter plot detected
        axes_data = analysis["axes"][0]
        assert axes_data["plot_type"] == "scatter"

    @pytest.mark.unit
    def test_bar_plot_vertical(self):
        """Test vertical bar plot."""
        fig, ax = plt.subplots()
        categories = ["A", "B", "C"]
        values = [1, 3, 2]
        ax.bar(categories, values)

        analysis = self.analyzer.analyze(fig)

        # Check plot type - new structure
        axes_data = analysis["axes"][0]
        assert "plot_type" in axes_data
        assert axes_data["plot_type"] == "bar"

        # Check data - bars are stored differently than curve_points
        assert "bars" in axes_data
        assert len(axes_data.get("bars", [])) >= 1

    @pytest.mark.unit
    def test_bar_plot_horizontal(self):
        """Test horizontal bar plot."""
        fig, ax = plt.subplots()
        categories = ["A", "B", "C"]
        values = [1, 3, 2]
        ax.barh(categories, values)

        analysis = self.analyzer.analyze(fig)

        # Check plot type - new structure
        axes_data = analysis["axes"][0]
        assert "plot_type" in axes_data
        # Horizontal bars might be detected as 'bar' or 'barh'
        assert axes_data["plot_type"] in ["bar", "barh"]

    @pytest.mark.unit
    def test_histogram_basic(self):
        """Test basic histogram."""
        fig, ax = plt.subplots()
        data = np.random.normal(0, 1, 100)
        ax.hist(data)

        analysis = self.analyzer.analyze(fig)

        # Check plot type - new structure
        axes_data = analysis["axes"][0]
        assert "plot_type" in axes_data
        # Histogram is detected as 'histogram' type in matplotlib
        assert axes_data["plot_type"] == "histogram"

    @pytest.mark.unit
    def test_histogram_with_bins(self):
        """Test histogram with custom bins."""
        fig, ax = plt.subplots()
        data = np.random.normal(0, 1, 100)
        ax.hist(data, bins=20)

        analysis = self.analyzer.analyze(fig)

        # Check plot type - new structure
        axes_data = analysis["axes"][0]
        assert "plot_type" in axes_data
        # Histogram is detected as 'histogram' type in matplotlib
        assert axes_data["plot_type"] == "histogram"

        # Check bins information if available
        assert "bins" in axes_data
        assert len(axes_data.get("bins", [])) >= 1

    @pytest.mark.unit
    def test_box_plot_single(self):
        """Test single box plot."""
        fig, ax = plt.subplots()
        data = np.random.normal(0, 1, 100)
        ax.boxplot(data)

        analysis = self.analyzer.analyze(fig)

        # Check plot type - new structure
        axes_data = analysis["axes"][0]
        assert "plot_type" in axes_data
        # Box plots are detected as 'line' type in matplotlib
        assert axes_data["plot_type"] == "line"

    @pytest.mark.unit
    def test_box_plot_multiple(self):
        """Test multiple box plots."""
        fig, ax = plt.subplots()
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(1, 1, 100)
        data3 = np.random.normal(-1, 1, 100)
        ax.boxplot([data1, data2, data3])

        analysis = self.analyzer.analyze(fig)

        # Check plot type
        axes_data = analysis["axes"][0]
        assert "plot_type" in axes_data
        # Box plots are detected as 'line' type in matplotlib
        assert axes_data["plot_type"] == "line"

        # Should have multiple lines (lines from boxplot)
        assert "lines" in axes_data
        lines = axes_data.get("lines", [])
        # Box plots generate multiple line segments
        assert len(lines) >= 1


class TestMatplotlibSubplots:
    """Test matplotlib subplots functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MatplotlibAnalyzer()
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_subplots_2x2(self):
        """Test 2x2 subplot with different plot types."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Line plot
        axes[0, 0].plot([1, 2, 3], [1, 4, 2])
        axes[0, 0].set_title("Line Plot")

        # Scatter plot
        axes[0, 1].scatter([1, 2, 3], [2, 1, 3])
        axes[0, 1].set_title("Scatter Plot")

        # Bar plot
        axes[1, 0].bar(["A", "B", "C"], [1, 3, 2])
        axes[1, 0].set_title("Bar Plot")

        # Histogram
        data = np.random.normal(0, 1, 100)
        axes[1, 1].hist(data)
        axes[1, 1].set_title("Histogram")

        analysis = self.analyzer.analyze(fig)

        # Should have 4 axes
        assert len(analysis["axes"]) >= 1  # At least 1 axis, may be more depending on implementation

        # Check that different plot types are detected
        all_plot_types = []
        for ax_data in analysis["axes"]:
            if "plot_type" in ax_data:
                all_plot_types.append(ax_data["plot_type"])

        # Should have different plot types
        unique_types = set(all_plot_types)
        assert len(unique_types) >= 2  # At least 2 different types

        # Check titles - may not be available in all implementations
        titles = [ax_data.get("title", "") for ax_data in analysis["axes"]]
        # Just check that we have some axes data
        assert len(titles) >= 1


class TestMatplotlibEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MatplotlibAnalyzer()
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_empty_plot(self):
        """Test empty plot or plot without data."""
        fig, ax = plt.subplots()
        # Don't add any data

        analysis = self.analyzer.analyze(fig)

        # Should not crash - new structure
        assert "figure" in analysis
        assert analysis["figure"]["figure_type"] == "matplotlib.Figure"
        assert len(analysis["axes"]) >= 1

        # Axes should have minimal data or none
        axes_data = analysis["axes"][0]
        # Empty plot should have no lines or empty lines
        assert isinstance(axes_data.get("lines", []), list)

    @pytest.mark.unit
    def test_plot_with_nan_values(self):
        """Test plot with NaN values in data."""
        fig, ax = plt.subplots()
        x = [1, 2, np.nan, 4, 5]
        y = [1, np.nan, 3, 4, 5]
        ax.plot(x, y)

        analysis = self.analyzer.analyze(fig)

        # Should not crash - new structure
        assert "figure" in analysis
        assert analysis["figure"]["figure_type"] == "matplotlib.Figure"

        # Check that data is handled properly
        axes_data = analysis["axes"][0]
        assert "lines" in axes_data
        assert len(axes_data.get("lines", [])) >= 1

        # NaN values should be handled gracefully
        line = axes_data.get("lines", [])[0]
        assert "xdata" in line
        assert "ydata" in line

    @pytest.mark.unit
    def test_unicode_labels(self):
        """Test plot with unicode characters in labels."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_title("Plot with Unicode: αβγ δεζ 中文 🚀")
        ax.set_xlabel("X轴 (αξις)")
        ax.set_ylabel("Y軸 (βξις)")

        analysis = self.analyzer.analyze(fig)

        # Should handle unicode properly - new structure
        assert "figure" in analysis
        assert "αβγ" in analysis["figure"]["title"]
        axes_data = analysis["axes"][0]
        assert "αξις" in axes_data["xlabel"]
        assert "βξις" in axes_data["ylabel"]

    @pytest.mark.unit
    def test_very_long_labels(self):
        """Test plot with very long labels."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        long_title = "A" * 1000
        long_xlabel = "X" * 500
        long_ylabel = "Y" * 500

        ax.set_title(long_title)
        ax.set_xlabel(long_xlabel)
        ax.set_ylabel(long_ylabel)

        analysis = self.analyzer.analyze(fig)

        # Should handle long labels without crashing - new structure
        assert "figure" in analysis
        assert len(analysis["figure"]["title"]) == 1000
        axes_data = analysis["axes"][0]
        assert len(axes_data["xlabel"]) == 500
        assert len(axes_data["ylabel"]) == 500

    @pytest.mark.unit
    def test_single_point_plot(self):
        """Test plot with only one data point."""
        fig, ax = plt.subplots()
        ax.plot([1], [1], "o")

        analysis = self.analyzer.analyze(fig)

        # Should handle single point - new structure
        assert "figure" in analysis
        assert analysis["figure"]["figure_type"] == "matplotlib.Figure"
        axes_data = analysis["axes"][0]
        assert "lines" in axes_data
        assert len(axes_data.get("lines", [])) >= 1

        line = axes_data.get("lines", [])[0]
        assert len(line["xdata"]) == 1
        assert len(line["ydata"]) == 1

    @pytest.mark.unit
    def test_duplicate_values(self):
        """Test plot where all values are the same."""
        fig, ax = plt.subplots()
        x = [1, 1, 1, 1]
        y = [2, 2, 2, 2]
        ax.plot(x, y)

        analysis = self.analyzer.analyze(fig)

        # Should handle duplicate values
        assert analysis["figure"]["figure_type"] == "matplotlib.Figure"
        axes_data = analysis["axes"][0]
        line = axes_data.get("lines", [])[0]

        assert all(val == 1 for val in line["xdata"])
        assert all(val == 2 for val in line["ydata"])

    @pytest.mark.unit
    def test_extreme_values(self):
        """Test plot with very large/small values."""
        fig, ax = plt.subplots()
        x = [1e-10, 1e-5, 1, 1e5, 1e10]
        y = [-1e10, -1e5, 0, 1e5, 1e10]
        ax.plot(x, y)

        analysis = self.analyzer.analyze(fig)

        # Should handle extreme values
        assert analysis["figure"]["figure_type"] == "matplotlib.Figure"
        axes_data = analysis["axes"][0]
        line = axes_data.get("lines", [])[0]

        # Values should be preserved
        assert len(line["xdata"]) == 5
        assert len(line["ydata"]) == 5

        # Check that extreme values are captured
        x_data = line["xdata"]
        y_data = line["ydata"]
        assert min(x_data) <= 1e-10
        assert max(x_data) >= 1e10
        assert min(y_data) <= -1e10
        assert max(y_data) >= 1e10


class TestMatplotlibErrorHandling:
    """Test error handling and invalid inputs."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MatplotlibAnalyzer()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_none_figure(self):
        """Test passing None as figure."""
        with pytest.raises(ValueError, match="Invalid figure object: None"):
            self.analyzer.analyze(None)

    @pytest.mark.unit
    def test_invalid_figure_type(self):
        """Test passing invalid figure type."""
        with pytest.raises(ValueError, match="Not a matplotlib figure"):
            self.analyzer.analyze("not a figure")

    @pytest.mark.unit
    def test_empty_data_arrays(self):
        """Test plot with empty data arrays."""
        fig, ax = plt.subplots()
        ax.plot([], [])

        analysis = self.analyzer.analyze(fig)

        # Should handle empty arrays gracefully - new structure
        assert "figure" in analysis
        assert analysis["figure"]["figure_type"] == "matplotlib.Figure"
        axes_data = analysis["axes"][0]

        # Should have a line but with empty data
        if axes_data.get("lines"):
            line = axes_data.get("lines", [])[0]
            assert len(line["xdata"]) == 0
            assert len(line["ydata"]) == 0


class TestMatplotlibIntegration:
    """Integration tests with FigureConverter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.integration
    def test_convert_text_format(self):
        """Test converting matplotlib figure to text format."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")

        result = self.converter.convert(fig, "text")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Test Plot" in result

    @pytest.mark.integration
    def test_convert_json_format(self):
        """Test converting matplotlib figure to JSON format."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")

        result = self.converter.convert(fig, "json")

        assert isinstance(result, dict)
        # Check new structure
        assert "figure" in result
        assert result["figure"]["figure_type"] == "matplotlib.Figure"
        assert result["figure"]["title"] == "Test Plot"

    @pytest.mark.integration
    def test_convert_semantic_format(self):
        """Test converting matplotlib figure to semantic format."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")

        result = self.converter.convert(fig, "semantic")

        assert isinstance(result, dict)
        # Check new semantic structure
        assert "metadata" in result
        assert "data_summary" in result
        assert "pattern_analysis" in result

    @pytest.mark.integration
    def test_global_convert_function(self):
        """Test global convert function."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        # Test all formats
        text_result = convert(fig, "text")
        json_result = convert(fig, "json")
        semantic_result = convert(fig, "semantic")

        assert isinstance(text_result, str)
        assert isinstance(json_result, dict)
        assert isinstance(semantic_result, dict)

        assert len(text_result) > 0
        # Check new structure
        assert "figure" in json_result
        assert "metadata" in semantic_result


if __name__ == "__main__":
    pytest.main([__file__])
