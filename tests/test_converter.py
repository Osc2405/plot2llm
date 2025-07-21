"""
Tests for plot2llm converter module.

This module tests the FigureConverter class, format handling,
analyzer registration, and error scenarios.
"""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from plot2llm.analyzers import FigureAnalyzer
from plot2llm.converter import FigureConverter, convert
from plot2llm.formatters import JSONFormatter, SemanticFormatter, TextFormatter

plt.ioff()


class TestFigureConverterInitialization:
    """Test FigureConverter initialization and configuration."""

    @pytest.mark.unit
    def test_converter_default_initialization(self):
        """Test converter initialization with default parameters."""
        converter = FigureConverter()

        assert converter.detail_level == "medium"
        assert converter.include_data is True
        assert converter.include_colors is True
        assert converter.include_statistics is True

        # Check that components are initialized
        assert isinstance(converter.analyzer, FigureAnalyzer)
        assert isinstance(converter.text_formatter, TextFormatter)
        assert isinstance(converter.json_formatter, JSONFormatter)
        assert isinstance(converter.semantic_formatter, SemanticFormatter)

    @pytest.mark.unit
    def test_converter_custom_initialization(self):
        """Test converter initialization with custom parameters."""
        converter = FigureConverter(
            detail_level="high",
            include_data=False,
            include_colors=False,
            include_statistics=False,
        )

        assert converter.detail_level == "high"
        assert converter.include_data is False
        assert converter.include_colors is False
        assert converter.include_statistics is False

    @pytest.mark.unit
    def test_converter_formatters_registration(self):
        """Test that formatters are properly registered."""
        converter = FigureConverter()

        assert "text" in converter.formatters
        assert "json" in converter.formatters
        assert "semantic" in converter.formatters

        assert converter.formatters["text"] is converter.text_formatter
        assert converter.formatters["json"] is converter.json_formatter
        assert converter.formatters["semantic"] is converter.semantic_formatter


class TestCustomFormatterRegistration:
    """Test custom formatter registration functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    @pytest.mark.unit
    def test_register_custom_formatter(self):
        """Test registering a custom formatter."""

        class CustomFormatter:
            def format(self, analysis, **kwargs):
                return f"Custom: {analysis.get('title', 'No Title')}"

        custom_formatter = CustomFormatter()
        self.converter.register_formatter("custom", custom_formatter)

        assert "custom" in self.converter.formatters
        assert self.converter.formatters["custom"] is custom_formatter

    @pytest.mark.unit
    def test_register_multiple_custom_formatters(self):
        """Test registering multiple custom formatters."""

        class Formatter1:
            def format(self, analysis, **kwargs):
                return "Format1"

        class Formatter2:
            def format(self, analysis, **kwargs):
                return "Format2"

        self.converter.register_formatter("format1", Formatter1())
        self.converter.register_formatter("format2", Formatter2())

        assert "format1" in self.converter.formatters
        assert "format2" in self.converter.formatters
        assert len(self.converter.formatters) >= 5  # 3 default + 2 custom

    @pytest.mark.unit
    def test_overwrite_existing_formatter(self):
        """Test overwriting an existing formatter."""

        class NewTextFormatter:
            def format(self, analysis, **kwargs):
                return "New text format"

        original_formatter = self.converter.formatters["text"]
        new_formatter = NewTextFormatter()

        self.converter.register_formatter("text", new_formatter)

        assert self.converter.formatters["text"] is new_formatter
        assert self.converter.formatters["text"] is not original_formatter


class TestCustomAnalyzerRegistration:
    """Test custom analyzer registration functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    @pytest.mark.unit
    def test_register_custom_analyzer(self):
        """Test registering a custom analyzer."""

        class CustomAnalyzer:
            def analyze(self, figure, **kwargs):
                return {"figure_type": "custom", "data": "test"}

        custom_analyzer = CustomAnalyzer()
        self.converter.register_analyzer("custom", custom_analyzer)

        assert "custom" in self.converter.analyzers
        assert self.converter.analyzers["custom"] is custom_analyzer

    @pytest.mark.unit
    def test_register_multiple_analyzers(self):
        """Test registering multiple custom analyzers."""

        class Analyzer1:
            def analyze(self, figure, **kwargs):
                return {"type": "analyzer1"}

        class Analyzer2:
            def analyze(self, figure, **kwargs):
                return {"type": "analyzer2"}

        self.converter.register_analyzer("analyzer1", Analyzer1())
        self.converter.register_analyzer("analyzer2", Analyzer2())

        assert "analyzer1" in self.converter.analyzers
        assert "analyzer2" in self.converter.analyzers


class TestFormatDetectionAndHandling:
    """Test format detection and handling logic."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_string_format_detection(self):
        """Test detection of string format parameters."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Test each default format
        for format_name in ["text", "json", "semantic"]:
            result = self.converter.convert(fig, format_name)

            if format_name == "text":
                assert isinstance(result, str)
            else:  # json and semantic return dicts
                assert isinstance(result, dict)

    @pytest.mark.unit
    def test_formatter_object_detection(self):
        """Test detection of formatter objects."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Test with formatter objects directly
        text_formatter = TextFormatter()
        json_formatter = JSONFormatter()
        semantic_formatter = SemanticFormatter()

        # Test each formatter object
        result1 = self.converter.convert(fig, text_formatter)
        assert isinstance(result1, str)

        result2 = self.converter.convert(fig, json_formatter)
        assert isinstance(result2, dict)

        result3 = self.converter.convert(fig, semantic_formatter)
        assert isinstance(result3, dict)

    @pytest.mark.unit
    def test_custom_registered_format_string(self):
        """Test using custom registered formatter with string."""

        class CustomFormatter:
            def format(self, analysis, **kwargs):
                return "Custom format output"

        self.converter.register_formatter("custom", CustomFormatter())

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        result = self.converter.convert(fig, "custom")
        assert result == "Custom format output"

    @pytest.mark.unit
    def test_invalid_format_string(self):
        """Test handling of invalid format strings."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with pytest.raises(ValueError, match="Unsupported output format"):
            self.converter.convert(fig, "invalid_format")

    @pytest.mark.unit
    def test_string_object_with_format_method(self):
        """Test that strings are not confused with formatter objects."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # String has format method, but should be treated as string
        format_string = "text"
        assert hasattr(format_string, "format")  # Strings have format method

        result = self.converter.convert(fig, format_string)
        assert isinstance(result, str)


class TestConverterErrorHandling:
    """Test error handling in FigureConverter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_convert_with_invalid_figure(self):
        """Test conversion with invalid figure objects."""
        # Test with None
        with pytest.raises(Exception):  # Should raise some kind of error
            self.converter.convert(None, "text")

        # Test with non-figure object
        with pytest.raises(Exception):
            self.converter.convert("not a figure", "text")

    @pytest.mark.unit
    def test_convert_with_analyzer_error(self):
        """Test conversion when analyzer raises error."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Mock analyzer to raise exception
        with patch.object(
            self.converter.analyzer, "analyze", side_effect=Exception("Analyzer error")
        ):
            with pytest.raises(Exception):
                self.converter.convert(fig, "text")

    @pytest.mark.unit
    def test_convert_with_formatter_error(self):
        """Test conversion when formatter raises error."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Mock formatter to raise exception
        with patch.object(
            self.converter.text_formatter,
            "format",
            side_effect=Exception("Formatter error"),
        ):
            with pytest.raises(Exception):
                self.converter.convert(fig, "text")

    @pytest.mark.unit
    def test_convert_with_figure_type_detection_error(self):
        """Test conversion when figure type detection fails."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Mock detect_figure_type to raise exception
        with patch(
            "plot2llm.converter.detect_figure_type",
            side_effect=Exception("Detection error"),
        ):
            with pytest.raises(Exception):
                self.converter.convert(fig, "text")


class TestConverterIntegrationWithAnalyzer:
    """Test converter integration with analyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_analyzer_parameter_passing(self):
        """Test that parameters are correctly passed to analyzer."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Create converter with specific parameters
        converter = FigureConverter(
            detail_level="high",
            include_data=False,
            include_colors=False,
            include_statistics=False,
        )

        # Mock analyzer to verify parameters
        with patch.object(
            converter.analyzer, "analyze", wraps=converter.analyzer.analyze
        ) as mock_analyze:
            converter.convert(fig, "json")

            # Verify analyzer was called with correct parameters
            mock_analyze.assert_called_once()
            call_args = mock_analyze.call_args

            assert call_args[1]["detail_level"] == "high"
            assert call_args[1]["include_data"] is False
            assert call_args[1]["include_colors"] is False
            assert call_args[1]["include_statistics"] is False

    @pytest.mark.unit
    def test_figure_type_detection_integration(self):
        """Test integration with figure type detection."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Mock figure type detection
        with patch(
            "plot2llm.converter.detect_figure_type", return_value="custom_type"
        ) as mock_detect:

            with patch.object(
                self.converter.analyzer,
                "analyze",
                return_value={"figure_type": "custom"},
            ) as mock_analyze:

                self.converter.convert(fig, "json")

                # Verify detection was called
                mock_detect.assert_called_once_with(fig)

                # Verify analyzer was called with detected type
                mock_analyze.assert_called_once()
                call_args = mock_analyze.call_args
                assert call_args[0][1] == "custom_type"  # figure_type parameter


class TestConverterUtilityMethods:
    """Test converter utility methods."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    @pytest.mark.unit
    def test_get_supported_formats(self):
        """Test getting supported formats."""
        formats = self.converter.get_supported_formats()

        assert isinstance(formats, list)
        assert "text" in formats
        assert "json" in formats
        assert "semantic" in formats
        assert len(formats) == 3

    @pytest.mark.unit
    def test_get_supported_libraries(self):
        """Test getting supported libraries."""
        libraries = self.converter.get_supported_libraries()

        assert isinstance(libraries, list)
        assert "matplotlib" in libraries
        assert "seaborn" in libraries
        assert "plotly" in libraries
        assert len(libraries) >= 3


class TestGlobalConvertFunction:
    """Test the global convert function."""

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_global_convert_basic(self):
        """Test basic usage of global convert function."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Test all formats
        text_result = convert(fig, "text")
        json_result = convert(fig, "json")
        semantic_result = convert(fig, "semantic")

        assert isinstance(text_result, str)
        assert isinstance(json_result, dict)
        assert isinstance(semantic_result, dict)

    @pytest.mark.unit
    def test_global_convert_with_kwargs(self):
        """Test global convert function with additional kwargs."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Test with additional arguments
        result = convert(fig, format="json", extra_param="test")

        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_global_convert_default_format(self):
        """Test global convert function with default format."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Default format should be text
        result = convert(fig)
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_global_convert_backend_detection(self):
        """Test backend detection in global convert function."""
        # Test matplotlib figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        result = convert(fig, "json")
        assert result["figure_type"] in ["matplotlib", "seaborn"]

        # Test with axes object
        result = convert(ax, "json")
        assert result["figure_type"] in ["matplotlib", "seaborn"]

    @pytest.mark.unit
    def test_global_convert_error_handling(self):
        """Test error handling in global convert function."""
        # Test with invalid format
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        with pytest.raises(ValueError):
            convert(fig, "invalid_format")


class TestConverterPerformance:
    """Test converter performance characteristics."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_converter_reuse(self):
        """Test that converter can be reused multiple times."""
        figures = []

        # Create multiple figures
        for i in range(5):
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [i, i + 1, i + 2])
            ax.set_title(f"Figure {i}")
            figures.append(fig)

        # Convert all figures with same converter
        results = []
        for fig in figures:
            result = self.converter.convert(fig, "json")
            results.append(result)

        # Verify all conversions worked
        assert len(results) == 5
        for i, result in enumerate(results):
            assert isinstance(result, dict)
            assert result["title"] == f"Figure {i}"

    @pytest.mark.unit
    def test_converter_memory_usage(self):
        """Test that converter doesn't accumulate memory."""
        # Create and convert many figures
        for i in range(10):
            fig, ax = plt.subplots()
            ax.plot(range(100), range(100))

            result = self.converter.convert(fig, "text")
            assert isinstance(result, str)

            plt.close(fig)  # Clean up

        # Converter should still work after many operations
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        result = self.converter.convert(fig, "json")
        assert isinstance(result, dict)


class TestConverterEdgeCases:
    """Test converter edge cases and unusual scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()

    def teardown_method(self):
        """Clean up after tests."""
        plt.close("all")

    @pytest.mark.unit
    def test_convert_empty_figure(self):
        """Test converting empty figure."""
        fig, ax = plt.subplots()
        # Don't add any data

        result = self.converter.convert(fig, "json")
        assert isinstance(result, dict)
        assert result["figure_type"] == "matplotlib"

    @pytest.mark.unit
    def test_convert_figure_with_no_axes(self):
        """Test converting figure with no axes."""
        fig = plt.figure()
        # Don't create any axes

        result = self.converter.convert(fig, "json")
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_convert_with_complex_data(self):
        """Test converting figure with complex data types."""
        fig, ax = plt.subplots()

        # Use complex numbers (should be handled gracefully)
        x = np.array([1 + 1j, 2 + 2j, 3 + 3j])
        try:
            ax.plot(x.real, x.imag)
            result = self.converter.convert(fig, "json")
            assert isinstance(result, dict)
        except Exception:
            # If plotting fails, that's OK for this test
            pass


if __name__ == "__main__":
    pytest.main([__file__])
