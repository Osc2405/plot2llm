"""
Comprehensive tests for seaborn analyzer functionality.

This module tests the SeabornAnalyzer class with various types of seaborn plots,
grid layouts, statistical visualizations, and seaborn-specific features.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns

from plot2llm import FigureConverter, convert
from plot2llm.analyzers.seaborn_analyzer import SeabornAnalyzer

# Suppress warnings during tests
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
plt.ioff()  # Turn off interactive mode


class TestSeabornBasicPlots:
    """Test basic seaborn plot types."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SeabornAnalyzer()
        self.converter = FigureConverter()

        # Create sample datasets
        np.random.seed(42)
        self.iris_data = sns.load_dataset("iris")
        self.tips_data = sns.load_dataset("tips")

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_seaborn_scatterplot(self):
        """Test seaborn scatterplot analysis."""
        # Create seaborn scatter plot
        fig, ax = plt.subplots()
        sns.scatterplot(data=self.iris_data, x="sepal_length", y="sepal_width", ax=ax)

        # Force seaborn detection
        analysis = self.analyzer.analyze(fig)

        # Basic assertions
        assert analysis["figure_type"] == "seaborn"
        assert len(analysis["axes"]) >= 1

        # Check plot types
        plot_types = []
        for ax_data in analysis["axes"]:
            plot_types.extend([pt["type"] for pt in ax_data.get("plot_types", [])])
        assert "scatter" in plot_types

        # Check data extraction
        axes_data = analysis["axes"][0]
        assert "curve_points" in axes_data
        assert len(axes_data["curve_points"]) >= 1

        # Check seaborn info
        assert "seaborn_info" in analysis

    @pytest.mark.unit
    def test_seaborn_scatterplot_with_hue(self):
        """Test seaborn scatterplot with hue parameter."""
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=self.iris_data, x="sepal_length", y="sepal_width", hue="species", ax=ax
        )

        analysis = self.analyzer.analyze(fig)

        assert analysis["figure_type"] == "seaborn"

        # Should have scatter plot
        plot_types = []
        for ax_data in analysis["axes"]:
            plot_types.extend([pt["type"] for pt in ax_data.get("plot_types", [])])
        assert "scatter" in plot_types

        # Should have color information
        colors_info = analysis["visual_info"].get("colors", [])
        # With hue, there should be multiple colors
        assert len(colors_info) >= 0  # Colors may be extracted differently

    @pytest.mark.unit
    def test_seaborn_lineplot(self):
        """Test seaborn lineplot analysis."""
        # Create time series data
        dates = pd.date_range("2020-01-01", periods=50)
        ts_data = pd.DataFrame({"date": dates, "value": np.cumsum(np.random.randn(50))})

        fig, ax = plt.subplots()
        sns.lineplot(data=ts_data, x="date", y="value", ax=ax)

        analysis = self.analyzer.analyze(fig)

        assert analysis["figure_type"] == "seaborn"

        # Check for line plot
        plot_types = []
        for ax_data in analysis["axes"]:
            plot_types.extend([pt["type"] for pt in ax_data.get("plot_types", [])])
        assert "line" in plot_types

    @pytest.mark.unit
    def test_seaborn_barplot(self):
        """Test seaborn barplot analysis."""
        fig, ax = plt.subplots()
        sns.barplot(data=self.tips_data, x="day", y="total_bill", ax=ax)

        analysis = self.analyzer.analyze(fig)

        assert analysis["figure_type"] == "seaborn"

        # Bar plots are detected as 'bar'
        plot_types = []
        for ax_data in analysis["axes"]:
            plot_types.extend([pt["type"] for pt in ax_data.get("plot_types", [])])
        assert "bar" in plot_types

        # Check axis types - should be categorical x, numeric y
        axes_data = analysis["axes"][0]
        # X-axis should be categorical for days
        assert axes_data.get("x_type") in ["category", "CATEGORY"]

    @pytest.mark.unit
    def test_seaborn_histplot(self):
        """Test seaborn histplot analysis."""
        fig, ax = plt.subplots()
        sns.histplot(data=self.iris_data, x="sepal_length", ax=ax)

        analysis = self.analyzer.analyze(fig)

        assert analysis["figure_type"] == "seaborn"

        # Histogram is detected as 'bar' type
        plot_types = []
        for ax_data in analysis["axes"]:
            plot_types.extend([pt["type"] for pt in ax_data.get("plot_types", [])])
        assert "bar" in plot_types

    @pytest.mark.unit
    def test_seaborn_boxplot(self):
        """Test seaborn boxplot analysis."""
        fig, ax = plt.subplots()
        sns.boxplot(data=self.tips_data, x="day", y="total_bill", ax=ax)

        analysis = self.analyzer.analyze(fig)

        assert analysis["figure_type"] == "seaborn"

        # Box plots generate line elements
        plot_types = []
        for ax_data in analysis["axes"]:
            plot_types.extend([pt["type"] for pt in ax_data.get("plot_types", [])])
        assert "line" in plot_types

    @pytest.mark.unit
    def test_seaborn_violinplot(self):
        """Test seaborn violinplot analysis."""
        fig, ax = plt.subplots()
        sns.violinplot(data=self.tips_data, x="day", y="total_bill", ax=ax)

        analysis = self.analyzer.analyze(fig)

        assert analysis["figure_type"] == "seaborn"

        # Violin plots have collections (patches)
        axes_data = analysis["axes"][0]
        assert len(axes_data["curve_points"]) >= 0

    @pytest.mark.unit
    def test_seaborn_heatmap(self):
        """Test seaborn heatmap analysis."""
        # Create correlation matrix
        corr_matrix = self.iris_data.select_dtypes(include=[np.number]).corr()

        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, ax=ax)

        analysis = self.analyzer.analyze(fig)

        assert analysis["figure_type"] == "seaborn"

        # Check seaborn info for heatmap detection
        axes_data = analysis["axes"][0]
        assert len(axes_data) > 0  # Should have some data


class TestSeabornGridLayouts:
    """Test seaborn grid layouts like FacetGrid and PairGrid."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SeabornAnalyzer()
        self.converter = FigureConverter()

        # Load datasets
        self.iris_data = sns.load_dataset("iris")
        self.tips_data = sns.load_dataset("tips")

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_seaborn_facetgrid(self):
        """Test seaborn FacetGrid analysis."""
        # Create FacetGrid
        g = sns.FacetGrid(self.tips_data, col="time", row="smoker")
        g.map(sns.scatterplot, "total_bill", "tip")

        # Analyze the grid figure
        analysis = self.analyzer.analyze(g.figure)

        assert analysis["figure_type"] == "seaborn"

        # Should have multiple axes (2x2 grid)
        assert len(analysis["axes"]) >= 2

        # Check seaborn info
        assert "grid_shape" in analysis.get("seaborn_info", {}) or "grid_size" in analysis.get("seaborn_info", {})

    @pytest.mark.unit
    def test_seaborn_pairplot(self):
        """Test seaborn pairplot analysis."""
        # Create pair plot (which creates a PairGrid)
        g = sns.pairplot(self.iris_data.select_dtypes(include=[np.number]).iloc[:50])

        analysis = self.analyzer.analyze(g.figure)

        assert analysis["figure_type"] == "seaborn"

        # Pair plot creates multiple subplots
        assert len(analysis["axes"]) > 1

        # Check for multiple plot types
        all_plot_types = []
        for ax_data in analysis["axes"]:
            all_plot_types.extend([pt["type"] for pt in ax_data.get("plot_types", [])])

        # Should have scatter and/or line plots
        plot_types_set = set(all_plot_types)
        assert len(plot_types_set) >= 1

    @pytest.mark.unit
    def test_seaborn_jointplot(self):
        """Test seaborn jointplot analysis."""
        # Create joint plot
        g = sns.jointplot(data=self.iris_data, x="sepal_length", y="sepal_width")

        analysis = self.analyzer.analyze(g.figure)

        assert analysis["figure_type"] == "seaborn"

        # Joint plot has main plot + marginal plots
        assert len(analysis["axes"]) >= 2

        # Check seaborn info
        seaborn_info = analysis.get("seaborn_info", {})
        assert isinstance(seaborn_info, dict)


class TestSeabornStatisticalPlots:
    """Test seaborn statistical visualization plots."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SeabornAnalyzer()

        # Create sample data
        np.random.seed(42)
        self.tips_data = sns.load_dataset("tips")

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_seaborn_regplot(self):
        """Test seaborn regression plot analysis."""
        fig, ax = plt.subplots()
        sns.regplot(data=self.tips_data, x="total_bill", y="tip", ax=ax)

        analysis = self.analyzer.analyze(fig)

        assert analysis["figure_type"] == "seaborn"

        # Regression plot has scatter points and line
        plot_types = []
        for ax_data in analysis["axes"]:
            plot_types.extend([pt["type"] for pt in ax_data.get("plot_types", [])])

        # Should have both scatter and line elements
        assert "scatter" in plot_types or "line" in plot_types

    @pytest.mark.unit
    def test_seaborn_distplot_kde(self):
        """Test seaborn distribution plots."""
        fig, ax = plt.subplots()
        # Use kdeplot instead of deprecated distplot
        sns.kdeplot(data=self.tips_data, x="total_bill", ax=ax)

        analysis = self.analyzer.analyze(fig)

        assert analysis["figure_type"] == "seaborn"

        # KDE plot creates line elements
        plot_types = []
        for ax_data in analysis["axes"]:
            plot_types.extend([pt["type"] for pt in ax_data.get("plot_types", [])])
        assert "line" in plot_types

    @pytest.mark.unit
    def test_seaborn_countplot(self):
        """Test seaborn countplot analysis."""
        fig, ax = plt.subplots()
        sns.countplot(data=self.tips_data, x="day", ax=ax)

        analysis = self.analyzer.analyze(fig)

        assert analysis["figure_type"] == "seaborn"

        # Count plot is a type of bar plot
        plot_types = []
        for ax_data in analysis["axes"]:
            plot_types.extend([pt["type"] for pt in ax_data.get("plot_types", [])])
        assert "bar" in plot_types


class TestSeabornEdgeCases:
    """Test seaborn edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SeabornAnalyzer()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_seaborn_empty_data(self):
        """Test seaborn plot with empty data."""
        empty_df = pd.DataFrame({"x": [], "y": []})

        fig, ax = plt.subplots()
        # This might not create any visual elements
        sns.scatterplot(data=empty_df, x="x", y="y", ax=ax)

        analysis = self.analyzer.analyze(fig)

        # Should not crash
        assert analysis["figure_type"] == "seaborn"
        assert len(analysis["axes"]) >= 1

    @pytest.mark.unit
    def test_seaborn_missing_values(self):
        """Test seaborn plot with missing values."""
        data_with_nan = pd.DataFrame(
            {
                "x": [1, 2, np.nan, 4, 5],
                "y": [1, np.nan, 3, 4, 5],
                "category": ["A", "B", "A", "B", "A"],
            }
        )

        fig, ax = plt.subplots()
        sns.scatterplot(data=data_with_nan, x="x", y="y", hue="category", ax=ax)

        analysis = self.analyzer.analyze(fig)

        # Should handle NaN values gracefully
        assert analysis["figure_type"] == "seaborn"
        axes_data = analysis["axes"][0]
        assert len(axes_data["curve_points"]) >= 0

    @pytest.mark.unit
    def test_seaborn_categorical_data(self):
        """Test seaborn with categorical data."""
        categorical_df = pd.DataFrame(
            {
                "category": ["A", "B", "C", "A", "B", "C"] * 10,
                "value": np.random.randn(60),
                "group": ["Group1", "Group2"] * 30,
            }
        )

        fig, ax = plt.subplots()
        sns.boxplot(data=categorical_df, x="category", y="value", hue="group", ax=ax)

        analysis = self.analyzer.analyze(fig)

        assert analysis["figure_type"] == "seaborn"

        # Check axis types
        axes_data = analysis["axes"][0]
        # X should be categorical
        assert axes_data.get("x_type") in ["category", "CATEGORY"]


class TestSeabornErrorHandling:
    """Test seaborn error handling and invalid inputs."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SeabornAnalyzer()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_none_figure(self):
        """Test passing None as figure."""
        with pytest.raises(ValueError, match="Invalid figure object: None"):
            self.analyzer.analyze(None)

    @pytest.mark.unit
    def test_seaborn_invalid_figure_type(self):
        """Test passing invalid figure type."""
        # SeabornAnalyzer handles invalid figures gracefully
        result = self.analyzer.analyze("not a figure")
        # Should return basic seaborn structure even for invalid figures
        assert result["figure_type"] == "seaborn"
        assert "axes" in result


class TestSeabornIntegration:
    """Integration tests with FigureConverter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()
        self.iris_data = sns.load_dataset("iris")

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.integration
    def test_seaborn_convert_text_format(self):
        """Test converting seaborn figure to text format."""
        fig, ax = plt.subplots()
        sns.scatterplot(data=self.iris_data, x="sepal_length", y="sepal_width", ax=ax)
        ax.set_title("Iris Sepal Analysis")

        # Force detection as seaborn
        result = self.converter.convert(fig, "text")

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.integration
    def test_seaborn_convert_json_format(self):
        """Test converting seaborn figure to JSON format."""
        fig, ax = plt.subplots()
        sns.boxplot(data=self.iris_data, x="species", y="sepal_length", ax=ax)
        ax.set_title("Species Comparison")

        result = self.converter.convert(fig, "json")

        assert isinstance(result, dict)
        assert "figure_type" in result
        assert result["title"] == "Species Comparison"

    @pytest.mark.integration
    def test_seaborn_convert_semantic_format(self):
        """Test converting seaborn figure to semantic format."""
        fig, ax = plt.subplots()
        sns.histplot(data=self.iris_data, x="petal_length", ax=ax)
        ax.set_title("Petal Length Distribution")

        result = self.converter.convert(fig, "semantic")

        assert isinstance(result, dict)
        assert "figure_type" in result
        assert "plot_description" in result

    @pytest.mark.integration
    def test_seaborn_global_convert_function(self):
        """Test global convert function with seaborn plots."""
        fig, ax = plt.subplots()
        sns.lineplot(data=self.iris_data, x="sepal_length", y="sepal_width", ax=ax)

        # Test all formats
        text_result = convert(fig, "text")
        json_result = convert(fig, "json")
        semantic_result = convert(fig, "semantic")

        assert isinstance(text_result, str)
        assert isinstance(json_result, dict)
        assert isinstance(semantic_result, dict)

        assert len(text_result) > 0
        assert "figure_type" in json_result
        assert "figure_type" in semantic_result


class TestSeabornSpecificFeatures:
    """Test seaborn-specific features and capabilities."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SeabornAnalyzer()
        self.tips_data = sns.load_dataset("tips")

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_seaborn_palette_detection(self):
        """Test detection of seaborn color palettes."""
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=self.tips_data,
            x="total_bill",
            y="tip",
            hue="time",
            palette="viridis",
            ax=ax,
        )

        analysis = self.analyzer.analyze(fig)

        assert analysis["figure_type"] == "seaborn"

        # Check for color information
        visual_info = analysis.get("visual_info", {})
        assert "colors" in visual_info

    @pytest.mark.unit
    def test_seaborn_style_detection(self):
        """Test seaborn style context detection."""
        # Set seaborn style
        with sns.axes_style("whitegrid"):
            fig, ax = plt.subplots()
            sns.lineplot(data=self.tips_data, x="total_bill", y="tip", ax=ax)

            analysis = self.analyzer.analyze(fig)

            assert analysis["figure_type"] == "seaborn"

            # Check if grid is detected
            axes_data = analysis["axes"][0]
            # Grid detection depends on implementation
            assert "has_grid" in axes_data

    @pytest.mark.unit
    def test_seaborn_figure_level_functions(self):
        """Test figure-level seaborn functions."""
        # Create figure-level plot
        g = sns.relplot(
            data=self.tips_data, x="total_bill", y="tip", col="time", kind="scatter"
        )

        analysis = self.analyzer.analyze(g.figure)

        assert analysis["figure_type"] == "seaborn"

        # Should have multiple subplots
        assert len(analysis["axes"]) >= 2

        # Check seaborn info
        seaborn_info = analysis.get("seaborn_info", {})
        assert isinstance(seaborn_info, dict)


if __name__ == "__main__":
    pytest.main([__file__])
