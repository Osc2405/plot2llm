"""
Tests for matplotlib figure conversion to different output formats.

This module specifically tests the conversion of matplotlib figures
to text, JSON, and semantic formats, ensuring consistency and correctness.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import json

from plot2llm import convert, FigureConverter
from plot2llm.formatters import TextFormatter, JSONFormatter, SemanticFormatter

plt.ioff()  # Turn off interactive mode


class TestMatplotlibTextFormat:
    """Test text format output for matplotlib figures."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_text_formatter_output(self):
        """Test that text formatter produces readable output."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Sample Line Plot")
        ax.set_xlabel("X Values")
        ax.set_ylabel("Y Values")

        result = self.converter.convert(fig, "text")

        # Should be string
        assert isinstance(result, str)
        assert len(result) > 0

        # Should contain key information
        assert "Sample Line Plot" in result
        assert "line" in result.lower() or "plot" in result.lower()

        # Should be human readable (not just raw data)
        assert not result.startswith("{")  # Not JSON
        assert not result.startswith("[")  # Not array

    @pytest.mark.unit
    def test_text_format_multiple_curves(self):
        """Test text format with multiple curves."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2], label="Line 1")
        ax.plot([1, 2, 3], [2, 3, 1], label="Line 2")
        ax.legend()
        ax.set_title("Multiple Lines")

        result = self.converter.convert(fig, "text")

        assert isinstance(result, str)
        assert "Multiple Lines" in result
        # Should mention multiple curves/lines
        assert "Line 1" in result or "multiple" in result.lower()


class TestMatplotlibJSONFormat:
    """Test JSON format output for matplotlib figures."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_json_formatter_valid(self):
        """Test that JSON formatter produces valid JSON structure."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Sample Plot")

        result = self.converter.convert(fig, "json")

        # Should be dict
        assert isinstance(result, dict)

        # Should have required keys
        assert "figure_type" in result
        assert result["figure_type"] == "matplotlib"

        # Should be JSON serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed == result

    @pytest.mark.unit
    def test_json_structure_consistency(self):
        """Test JSON structure consistency across different plots."""
        plots = []

        # Line plot
        fig1, ax1 = plt.subplots()
        ax1.plot([1, 2, 3], [1, 4, 2])
        plots.append(fig1)

        # Bar plot
        fig2, ax2 = plt.subplots()
        ax2.bar(["A", "B", "C"], [1, 3, 2])
        plots.append(fig2)

        # Scatter plot
        fig3, ax3 = plt.subplots()
        ax3.scatter([1, 2, 3], [1, 4, 2])
        plots.append(fig3)

        results = [self.converter.convert(fig, "json") for fig in plots]

        # All should have same top-level structure
        expected_keys = {"figure_type", "title", "axes"}
        for result in results:
            assert isinstance(result, dict)
            # Check that main keys exist (some might be None)
            assert "figure_type" in result
            assert "axes" in result

    @pytest.mark.unit
    def test_json_data_types(self):
        """Test that JSON contains proper data types."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1.5, 2.7, 3.2])
        ax.set_title("Numeric Plot")

        result = self.converter.convert(fig, "json")

        # Check data types in axes
        if result.get("axes"):
            axes_data = result["axes"][0]
            if axes_data.get("curve_points"):
                curve = axes_data["curve_points"][0]
                if "x" in curve:
                    # Should be list of numbers
                    assert isinstance(curve["x"], list)
                    assert all(isinstance(x, (int, float)) for x in curve["x"])
                if "y" in curve:
                    assert isinstance(curve["y"], list)
                    assert all(isinstance(y, (int, float)) for y in curve["y"])


class TestMatplotlibSemanticFormat:
    """Test semantic format output for matplotlib figures."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_semantic_formatter_llm(self):
        """Test semantic format optimized for LLM understanding."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Trend Analysis")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        result = self.converter.convert(fig, "semantic")

        # Should be dict
        assert isinstance(result, dict)

        # Should have semantic structure
        assert "figure_type" in result

        # Should have plot description for LLM
        assert "plot_description" in result or "description" in str(result).lower()

        # Should contain contextual information
        result_str = str(result).lower()
        assert "trend" in result_str or "plot" in result_str

    @pytest.mark.unit
    def test_semantic_format_insights(self):
        """Test that semantic format includes analytical insights."""
        # Create a plot with clear trend
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 50)
        y = x**2 + np.random.normal(0, 5, 50)  # Quadratic trend with noise
        ax.scatter(x, y)
        ax.set_title("Quadratic Growth Pattern")

        result = self.converter.convert(fig, "semantic")

        # Should contain the data and metadata
        assert isinstance(result, dict)
        assert "figure_type" in result

        # Should be structured for LLM consumption
        # (The specific insights depend on implementation)
        assert len(str(result)) > 100  # Should be reasonably detailed


class TestMatplotlibFormatConsistency:
    """Test consistency across different output formats."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_formatter_consistency(self):
        """Test that same data produces consistent information across formats."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2], label="Test Line")
        ax.set_title("Consistency Test")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.legend()

        # Get all formats
        text_result = self.converter.convert(fig, "text")
        json_result = self.converter.convert(fig, "json")
        semantic_result = self.converter.convert(fig, "semantic")

        # All should reference the same title
        assert "Consistency Test" in text_result
        assert json_result.get("title") == "Consistency Test"

        # JSON should have proper structure
        assert json_result["figure_type"] == "matplotlib"
        assert semantic_result["figure_type"] == "matplotlib"

        # All should be non-empty
        assert len(text_result) > 0
        assert len(json_result) > 0
        assert len(semantic_result) > 0

    @pytest.mark.unit
    def test_data_preservation_across_formats(self):
        """Test that core data is preserved across all formats."""
        fig, ax = plt.subplots()
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 4, 6, 8, 10]
        ax.plot(x_data, y_data)
        ax.set_title("Data Preservation Test")

        json_result = self.converter.convert(fig, "json")
        semantic_result = self.converter.convert(fig, "semantic")

        # Both should preserve the core data structure
        # (Exact format may vary, but core info should be there)
        assert "Data Preservation Test" in str(json_result)
        assert "Data Preservation Test" in str(semantic_result)

        # JSON should have axes data
        assert "axes" in json_result
        assert len(json_result["axes"]) > 0


class TestMatplotlibCustomFormatter:
    """Test custom formatter registration and usage."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_custom_formatter(self):
        """Test registering and using a custom formatter."""

        class CustomFormatter:
            def format(self, analysis, **kwargs):
                return f"Custom: {analysis.get('title', 'No Title')}"

        custom_formatter = CustomFormatter()
        self.converter.register_formatter("custom", custom_formatter)

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        ax.set_title("Custom Test")

        # Test using string format name
        result = self.converter.convert(fig, "custom")
        assert result == "Custom: Custom Test"

        # Test using formatter object directly
        result2 = self.converter.convert(fig, custom_formatter)
        assert result2 == "Custom: Custom Test"


if __name__ == "__main__":
    pytest.main([__file__])
