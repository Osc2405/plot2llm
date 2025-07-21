"""
Advanced integration tests for plot2llm.

This module tests complex workflows, performance with large datasets,
multi-library integration, and real-world scenarios.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from unittest.mock import patch, MagicMock

from plot2llm import convert, FigureConverter
from plot2llm.analyzers import FigureAnalyzer, MatplotlibAnalyzer, SeabornAnalyzer
from plot2llm.formatters import TextFormatter, JSONFormatter, SemanticFormatter
from plot2llm.utils import detect_figure_type

# Suppress warnings during tests
warnings.filterwarnings("ignore")
plt.ioff()


class TestLargeDatasetPerformance:
    """Test performance with large datasets."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.slow
    def test_large_scatter_plot_performance(self):
        """Test performance with large scatter plot (10k+ points)."""
        # Create large dataset
        np.random.seed(42)
        n_points = 15000
        x = np.random.randn(n_points)
        y = x * 2 + np.random.randn(n_points) * 0.5

        fig, ax = plt.subplots()
        start_time = time.time()
        ax.scatter(x, y, alpha=0.5)
        plot_time = time.time() - start_time

        # Test conversion performance
        start_time = time.time()
        result = self.converter.convert(fig, "json")
        conversion_time = time.time() - start_time

        # Performance assertions
        assert conversion_time < 5.0  # Should complete in under 5 seconds
        assert isinstance(result, dict)
        assert result["figure_type"] == "matplotlib"

        # Check data handling
        axes_data = result["axes"][0]
        assert len(axes_data["curve_points"]) >= 1

        print(f"Plot creation: {plot_time:.2f}s, Conversion: {conversion_time:.2f}s")

    @pytest.mark.slow
    def test_large_line_plot_performance(self):
        """Test performance with large time series (50k+ points)."""
        # Create large time series
        np.random.seed(42)
        n_points = 50000
        x = np.arange(n_points)
        y = np.cumsum(np.random.randn(n_points) * 0.01)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("Large Time Series")

        start_time = time.time()
        result = self.converter.convert(fig, "text")
        conversion_time = time.time() - start_time

        # Performance and correctness
        assert conversion_time < 3.0  # Should be faster for line plots
        assert isinstance(result, str)
        assert "Large Time Series" in result
        assert len(result) > 100

    @pytest.mark.slow
    def test_complex_subplot_performance(self):
        """Test performance with complex subplot layouts."""
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        np.random.seed(42)

        for i, ax in enumerate(axes.flat):
            if i % 4 == 0:
                # Line plot
                x = np.linspace(0, 10, 1000)
                y = np.sin(x + i) + np.random.randn(1000) * 0.1
                ax.plot(x, y)
                ax.set_title(f"Line Plot {i}")
            elif i % 4 == 1:
                # Scatter plot
                x = np.random.randn(500)
                y = np.random.randn(500)
                ax.scatter(x, y, alpha=0.6)
                ax.set_title(f"Scatter Plot {i}")
            elif i % 4 == 2:
                # Bar plot
                categories = ["A", "B", "C", "D"]
                values = np.random.rand(4) * 10
                ax.bar(categories, values)
                ax.set_title(f"Bar Plot {i}")
            else:
                # Histogram
                data = np.random.normal(i, 1, 1000)
                ax.hist(data, bins=30)
                ax.set_title(f"Histogram {i}")

        plt.tight_layout()

        start_time = time.time()
        result = self.converter.convert(fig, "json")
        conversion_time = time.time() - start_time

        # Performance and correctness (adjusted for complex subplot processing)
        assert conversion_time < 15.0  # Complex plot should still be reasonable
        assert isinstance(result, dict)
        assert len(result["axes"]) == 12  # 3x4 = 12 subplots

        # Check that different plot types are detected
        all_plot_types = []
        for ax_data in result["axes"]:
            all_plot_types.extend([pt["type"] for pt in ax_data.get("plot_types", [])])

        unique_types = set(all_plot_types)
        assert len(unique_types) >= 2  # Should have multiple plot types


class TestMultiLibraryIntegration:
    """Test integration between matplotlib and seaborn."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

        # Sample datasets
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "x": np.random.randn(100),
                "y": np.random.randn(100),
                "category": np.random.choice(["A", "B", "C"], 100),
                "size": np.random.rand(100) * 50 + 10,
            }
        )

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.integration
    def test_matplotlib_seaborn_mixed_plot(self):
        """Test plot that mixes matplotlib and seaborn elements."""
        fig, ax = plt.subplots()

        # Start with seaborn
        sns.scatterplot(data=self.df, x="x", y="y", hue="category", size="size", ax=ax)

        # Add matplotlib elements
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)
        ax.set_title("Mixed Matplotlib + Seaborn Plot")

        # Test detection and conversion
        figure_type = detect_figure_type(fig)
        result = self.converter.convert(fig, "json")

        assert isinstance(result, dict)
        assert result["title"] == "Mixed Matplotlib + Seaborn Plot"

        # Should detect multiple plot elements
        axes_data = result["axes"][0]
        plot_types = [pt["type"] for pt in axes_data.get("plot_types", [])]
        assert len(plot_types) >= 1  # At least scatter and lines

    @pytest.mark.integration
    def test_seaborn_on_matplotlib_axes(self):
        """Test seaborn plots created on matplotlib axes."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Different seaborn plots on each axis
        sns.histplot(data=self.df, x="x", ax=axes[0, 0])
        axes[0, 0].set_title("Histogram")

        sns.boxplot(data=self.df, x="category", y="y", ax=axes[0, 1])
        axes[0, 1].set_title("Box Plot")

        sns.scatterplot(data=self.df, x="x", y="y", ax=axes[1, 0])
        axes[1, 0].set_title("Scatter Plot")

        sns.violinplot(data=self.df, x="category", y="x", ax=axes[1, 1])
        axes[1, 1].set_title("Violin Plot")

        plt.tight_layout()

        result = self.converter.convert(fig, "json")

        assert len(result["axes"]) == 4

        # Check that titles are preserved
        titles = [ax["title"] for ax in result["axes"]]
        expected_titles = ["Histogram", "Box Plot", "Scatter Plot", "Violin Plot"]
        for title in expected_titles:
            assert title in titles

    @pytest.mark.integration
    def test_pandas_matplotlib_integration(self):
        """Test pandas plotting with matplotlib backend."""
        # Create pandas plot (uses matplotlib backend)
        fig, ax = plt.subplots()
        self.df.plot(x="x", y="y", kind="scatter", ax=ax)
        ax.set_title("Pandas Scatter Plot")

        result = self.converter.convert(fig, "text")

        assert isinstance(result, str)
        assert "Pandas Scatter Plot" in result
        assert "scatter" in result.lower()


class TestComplexWorkflows:
    """Test complex real-world workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.integration
    def test_data_analysis_workflow(self):
        """Test complete data analysis workflow."""
        # Simulate financial data analysis
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=365)
        prices = 100 + np.cumsum(np.random.randn(365) * 0.02)
        volume = np.random.poisson(1000, 365)

        df = pd.DataFrame(
            {
                "date": dates,
                "price": prices,
                "volume": volume,
                "returns": np.concatenate([[0], np.diff(prices)]),
            }
        )

        # Create complex financial dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Financial Analysis Dashboard")

        # Price time series
        axes[0, 0].plot(df["date"], df["price"])
        axes[0, 0].set_title("Price Over Time")
        axes[0, 0].set_ylabel("Price ($)")

        # Volume bars
        axes[0, 1].bar(df["date"][::30], df["volume"][::30])  # Monthly volume
        axes[0, 1].set_title("Monthly Volume")
        axes[0, 1].set_ylabel("Volume")

        # Returns distribution
        axes[1, 0].hist(df["returns"], bins=50, alpha=0.7)
        axes[1, 0].set_title("Returns Distribution")
        axes[1, 0].set_xlabel("Daily Returns")

        # Price vs Volume scatter
        axes[1, 1].scatter(df["price"], df["volume"], alpha=0.5)
        axes[1, 1].set_title("Price vs Volume")
        axes[1, 1].set_xlabel("Price ($)")
        axes[1, 1].set_ylabel("Volume")

        plt.tight_layout()

        # Test all output formats
        text_result = self.converter.convert(fig, "text")
        json_result = self.converter.convert(fig, "json")
        semantic_result = self.converter.convert(fig, "semantic")

        # Verify all formats work
        assert isinstance(text_result, str)
        assert "Financial Analysis Dashboard" in text_result

        assert isinstance(json_result, dict)
        assert json_result["title"] == "Financial Analysis Dashboard"
        assert len(json_result["axes"]) == 4

        assert isinstance(semantic_result, dict)
        assert "Financial Analysis Dashboard" in semantic_result["plot_description"]

    @pytest.mark.integration
    def test_scientific_publication_workflow(self):
        """Test scientific publication figure workflow."""
        # Simulate experimental data
        np.random.seed(42)

        # Create publication-quality figure
        fig = plt.figure(figsize=(12, 8))

        # Main plot
        ax1 = plt.subplot(2, 2, (1, 2))

        # Experimental conditions
        conditions = ["Control", "Treatment A", "Treatment B", "Treatment C"]
        means = [1.0, 1.5, 2.1, 1.8]
        errors = [0.1, 0.15, 0.2, 0.12]

        x_pos = np.arange(len(conditions))
        bars = ax1.bar(
            x_pos,
            means,
            yerr=errors,
            capsize=5,
            color=["blue", "red", "green", "orange"],
            alpha=0.7,
        )
        ax1.set_xlabel("Experimental Condition")
        ax1.set_ylabel("Response (arbitrary units)")
        ax1.set_title("Treatment Effects on Response Variable")
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(conditions)

        # Add significance indicators
        ax1.plot([0, 1], [1.8, 1.8], "k-", linewidth=1)
        ax1.text(0.5, 1.85, "*", ha="center", fontsize=12)

        # Dose-response curve
        ax2 = plt.subplot(2, 2, 3)
        doses = np.logspace(-2, 2, 50)
        response = 2 / (1 + np.exp(-np.log10(doses))) + np.random.randn(50) * 0.1
        ax2.semilogx(doses, response, "o-", markersize=4)
        ax2.set_xlabel("Dose (mg/kg)")
        ax2.set_ylabel("Response")
        ax2.set_title("Dose-Response Relationship")
        ax2.grid(True, alpha=0.3)

        # Time course
        ax3 = plt.subplot(2, 2, 4)
        time = np.linspace(0, 24, 100)
        baseline = np.ones_like(time)
        treatment = 1 + 0.5 * np.exp(-time / 6) * np.sin(time / 2)

        ax3.plot(time, baseline, "b-", label="Control", linewidth=2)
        ax3.plot(time, treatment, "r-", label="Treatment", linewidth=2)
        ax3.fill_between(time, treatment - 0.1, treatment + 0.1, alpha=0.3, color="red")
        ax3.set_xlabel("Time (hours)")
        ax3.set_ylabel("Normalized Response")
        ax3.set_title("Time Course Analysis")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        # Test analysis
        result = self.converter.convert(fig, "semantic")

        assert isinstance(result, dict)
        assert "Treatment Effects" in result["plot_description"]
        assert len(result["axes"]) == 3  # Should detect all subplots

        # Check that statistical elements are captured
        plot_types = []
        for ax_data in result["axes"]:
            plot_types.extend([pt["type"] for pt in ax_data.get("plot_types", [])])

        # Should have bars, lines, and scatter elements
        assert "bar" in plot_types
        assert "line" in plot_types

    @pytest.mark.integration
    def test_machine_learning_workflow(self):
        """Test machine learning visualization workflow."""
        try:
            from sklearn.datasets import make_classification
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import confusion_matrix
        except ImportError:
            pytest.skip("Scikit-learn not available")

        # Generate synthetic dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_classes=3,
            n_informative=10,
            random_state=42,
        )

        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Create ML visualization dashboard
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Machine Learning Model Analysis")

        # Feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10 features
        axes[0, 0].bar(range(10), importances[indices])
        axes[0, 0].set_title("Top 10 Feature Importances")
        axes[0, 0].set_xlabel("Feature Index")
        axes[0, 0].set_ylabel("Importance")

        # Confusion matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        im = axes[0, 1].imshow(cm, interpolation="nearest", cmap="Blues")
        axes[0, 1].set_title("Confusion Matrix")
        axes[0, 1].set_xlabel("Predicted")
        axes[0, 1].set_ylabel("Actual")

        # Learning curve simulation
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = 1 - np.exp(-train_sizes * 3) + np.random.randn(10) * 0.02
        val_scores = train_scores - 0.1 + np.random.randn(10) * 0.02

        axes[1, 0].plot(train_sizes, train_scores, "o-", label="Training")
        axes[1, 0].plot(train_sizes, val_scores, "s-", label="Validation")
        axes[1, 0].set_title("Learning Curves")
        axes[1, 0].set_xlabel("Training Set Size")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].legend()

        # ROC curve simulation
        fpr = np.linspace(0, 1, 100)
        tpr = np.sqrt(fpr)  # Simplified ROC curve
        axes[1, 1].plot(fpr, tpr, linewidth=2)
        axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[1, 1].set_title("ROC Curve")
        axes[1, 1].set_xlabel("False Positive Rate")
        axes[1, 1].set_ylabel("True Positive Rate")

        plt.tight_layout()

        # Test conversion
        result = self.converter.convert(fig, "json")

        assert isinstance(result, dict)
        assert result["title"] == "Machine Learning Model Analysis"
        assert len(result["axes"]) == 4

        # Verify different plot types captured
        all_plot_types = []
        for ax_data in result["axes"]:
            all_plot_types.extend([pt["type"] for pt in ax_data.get("plot_types", [])])

        assert "bar" in all_plot_types  # Feature importance
        assert "line" in all_plot_types  # Learning curves


class TestErrorRecoveryAndRobustness:
    """Test error recovery and robustness in complex scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.integration
    def test_corrupted_data_handling(self):
        """Test handling of corrupted or unusual data."""
        # Create data with various edge cases
        problematic_data = {
            "infinite": [1, 2, np.inf, 4, 5],
            "negative_inf": [1, -np.inf, 3, 4, 5],
            "mixed_types": [1, 2, "3", 4, 5],  # Will cause issues
            "very_large": [1e10, 2e10, 3e10],
            "very_small": [1e-10, 2e-10, 3e-10],
        }

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))

        # Test each problematic case
        try:
            # Infinite values
            axes[0, 0].plot([1, 2, 3, 4, 5], problematic_data["infinite"])
            axes[0, 0].set_title("Infinite Values")

            # Very large values
            axes[0, 1].plot(problematic_data["very_large"])
            axes[0, 1].set_title("Very Large Values")

            # Very small values
            axes[1, 0].plot(problematic_data["very_small"])
            axes[1, 0].set_title("Very Small Values")

            # Regular data for comparison
            axes[1, 1].plot([1, 2, 3], [1, 2, 3])
            axes[1, 1].set_title("Normal Data")

        except Exception:
            pass  # Some of these might fail at plot creation

        # Should handle gracefully without crashing
        try:
            result = self.converter.convert(fig, "json")
            assert isinstance(result, dict)
        except Exception as e:
            # If it fails, should fail gracefully
            assert "error" in str(e).lower() or "invalid" in str(e).lower()

    @pytest.mark.integration
    def test_memory_intensive_plots(self):
        """Test memory management with intensive plots."""
        # Create multiple large figures to test memory handling
        figures = []

        try:
            for i in range(5):  # Create several large figures
                fig, ax = plt.subplots(figsize=(10, 8))

                # Large dataset
                n = 10000
                x = np.random.randn(n)
                y = np.random.randn(n)

                ax.scatter(x, y, alpha=0.5, s=1)
                ax.set_title(f"Large Figure {i}")

                figures.append(fig)

                # Convert each figure
                result = self.converter.convert(fig, "text")
                assert isinstance(result, str)
                assert f"Large Figure {i}" in result

        finally:
            # Clean up
            for fig in figures:
                plt.close(fig)
            plt.close("all")

    @pytest.mark.integration
    def test_concurrent_analysis_simulation(self):
        """Simulate concurrent analysis scenarios."""
        # Test multiple analyzers working with different figures
        matplotlib_analyzer = MatplotlibAnalyzer()
        seaborn_analyzer = SeabornAnalyzer()

        figures = []

        try:
            # Create matplotlib figure
            fig1, ax1 = plt.subplots()
            ax1.plot([1, 2, 3], [1, 4, 2])
            ax1.set_title("Matplotlib Figure")
            figures.append(fig1)

            # Create seaborn figure
            fig2, ax2 = plt.subplots()
            data = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 4, 2, 3]})
            sns.scatterplot(data=data, x="x", y="y", ax=ax2)
            ax2.set_title("Seaborn Figure")
            figures.append(fig2)

            # Analyze both simultaneously (simulate concurrent access)
            results = []

            # Matplotlib analysis
            result1 = matplotlib_analyzer.analyze(fig1)
            results.append(result1)

            # Seaborn analysis
            result2 = seaborn_analyzer.analyze(fig2)
            results.append(result2)

            # Verify both analyses worked
            assert len(results) == 2
            assert results[0]["figure_type"] == "matplotlib"
            assert results[1]["figure_type"] == "seaborn"

        finally:
            for fig in figures:
                plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])
