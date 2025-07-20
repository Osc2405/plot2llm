"""
Tests for plot2llm base analyzer functionality.

This module tests the BaseAnalyzer abstract class and its default implementations.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch
from abc import ABC

from plot2llm.analyzers.base_analyzer import BaseAnalyzer

plt.ioff()


class ConcreteAnalyzer(BaseAnalyzer):
    """Concrete implementation of BaseAnalyzer for testing."""
    
    def analyze(self, figure, detail_level="medium", include_data=True, 
                include_colors=True, include_statistics=True):
        """Simple analyze implementation."""
        return {
            "figure_type": self._get_figure_type(figure),
            "dimensions": self._get_dimensions(figure),
            "title": self._get_title(figure),
            "axes_count": self._get_axes_count(figure)
        }
    
    def _get_figure_type(self, figure):
        """Get figure type implementation."""
        return "test_figure"
    
    def _get_dimensions(self, figure):
        """Get dimensions implementation."""
        return (8, 6)
    
    def _get_title(self, figure):
        """Get title implementation."""
        if hasattr(figure, '_suptitle') and figure._suptitle:
            return figure._suptitle.get_text()
        return "Test Title"
    
    def _get_axes(self, figure):
        """Get axes implementation."""
        if hasattr(figure, 'axes'):
            return figure.axes
        return []
    
    def _get_axes_count(self, figure):
        """Get axes count implementation."""
        return len(self._get_axes(figure))


class TestBaseAnalyzerAbstractMethods:
    """Test BaseAnalyzer abstract methods and interface."""
    
    @pytest.mark.unit
    def test_base_analyzer_is_abstract(self):
        """Test that BaseAnalyzer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAnalyzer()
    
    @pytest.mark.unit
    def test_concrete_analyzer_instantiation(self):
        """Test that concrete analyzer can be instantiated."""
        analyzer = ConcreteAnalyzer()
        assert isinstance(analyzer, BaseAnalyzer)
        assert analyzer.supported_types == []
    
    @pytest.mark.unit
    def test_concrete_analyzer_initialization(self):
        """Test concrete analyzer initialization."""
        analyzer = ConcreteAnalyzer()
        assert hasattr(analyzer, 'supported_types')
        assert isinstance(analyzer.supported_types, list)


class TestBaseAnalyzerMethods:
    """Test BaseAnalyzer concrete method implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ConcreteAnalyzer()
        
    def teardown_method(self):
        """Clean up after tests."""
        plt.close('all')
    
    @pytest.mark.unit
    def test_extract_basic_info(self):
        """Test extract_basic_info method."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        info = self.analyzer.extract_basic_info(fig)
        
        assert isinstance(info, dict)
        assert "figure_type" in info
        assert "dimensions" in info
        assert "title" in info
        assert "axes_count" in info
        
        assert info["figure_type"] == "test_figure"
        assert info["dimensions"] == (8, 6)
        assert info["axes_count"] == 1
    
    @pytest.mark.unit
    def test_extract_visual_info(self):
        """Test extract_visual_info method."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        # Set include_colors attribute for the test
        self.analyzer.include_colors = True
        
        info = self.analyzer.extract_visual_info(fig)
        
        assert isinstance(info, dict)
        # Should have these keys (even if empty due to default implementations)
        expected_keys = ["colors", "markers", "line_styles", "background_color"]
        for key in expected_keys:
            assert key in info
    
    @pytest.mark.unit
    def test_extract_visual_info_error_handling(self):
        """Test extract_visual_info error handling."""
        # Mock _get_colors to raise an exception
        with patch.object(self.analyzer, '_get_colors', side_effect=Exception("Test error")):
            self.analyzer.include_colors = True
            info = self.analyzer.extract_visual_info("invalid_figure")
            assert isinstance(info, dict)
            # Should return empty dict on error
    
    @pytest.mark.unit
    def test_default_implementations(self):
        """Test default implementations of optional methods."""
        fig, ax = plt.subplots()
        
        # Test default implementations return expected types
        assert isinstance(self.analyzer._get_axis_type(ax), str)
        assert self.analyzer._get_axis_type(ax) == "unknown"
        
        assert self.analyzer._get_x_label(ax) is None
        assert self.analyzer._get_y_label(ax) is None
        assert self.analyzer._get_x_range(ax) is None
        assert self.analyzer._get_y_range(ax) is None
        
        assert isinstance(self.analyzer._has_grid(ax), bool)
        assert self.analyzer._has_grid(ax) is False
        
        assert isinstance(self.analyzer._has_legend(ax), bool)
        assert self.analyzer._has_legend(ax) is False
        
        assert isinstance(self.analyzer._get_data_points(fig), int)
        assert self.analyzer._get_data_points(fig) == 0
        
        assert isinstance(self.analyzer._get_data_types(fig), list)
        assert self.analyzer._get_data_types(fig) == []
        
        assert isinstance(self.analyzer._get_statistics(fig), dict)
        assert self.analyzer._get_statistics(fig) == {}
        
        assert isinstance(self.analyzer._get_colors(fig), list)
        assert self.analyzer._get_colors(fig) == []
        
        assert isinstance(self.analyzer._get_markers(fig), list)
        assert self.analyzer._get_markers(fig) == []
        
        assert isinstance(self.analyzer._get_line_styles(fig), list)
        assert self.analyzer._get_line_styles(fig) == []
        
        assert self.analyzer._get_background_color(fig) is None
        assert self.analyzer._get_axis_title(ax) is None


class TestBaseAnalyzerWithRealFigures:
    """Test BaseAnalyzer with real matplotlib figures."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ConcreteAnalyzer()
        
    def teardown_method(self):
        """Clean up after tests."""
        plt.close('all')
    
    @pytest.mark.unit
    def test_analyze_simple_figure(self):
        """Test analyzing a simple matplotlib figure."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        ax.set_title("Test Plot")
        
        result = self.analyzer.analyze(fig)
        
        assert isinstance(result, dict)
        assert result["figure_type"] == "test_figure"
        assert result["dimensions"] == (8, 6)
        assert result["axes_count"] == 1
    
    @pytest.mark.unit
    def test_analyze_figure_with_suptitle(self):
        """Test analyzing figure with suptitle."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        fig.suptitle("Main Title")
        
        result = self.analyzer.analyze(fig)
        
        assert result["title"] == "Main Title"
    
    @pytest.mark.unit
    def test_analyze_subplot_figure(self):
        """Test analyzing figure with multiple subplots."""
        fig, axes = plt.subplots(2, 2)
        
        for i, ax in enumerate(axes.flat):
            ax.plot([1, 2, 3], [i, i+1, i+2])
            ax.set_title(f"Subplot {i}")
        
        result = self.analyzer.analyze(fig)
        
        assert result["axes_count"] == 4
    
    @pytest.mark.unit
    def test_analyze_with_options(self):
        """Test analyze method with different options."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        # Test with different parameter combinations
        result1 = self.analyzer.analyze(fig, detail_level="high")
        result2 = self.analyzer.analyze(fig, include_data=False)
        result3 = self.analyzer.analyze(fig, include_colors=False)
        result4 = self.analyzer.analyze(fig, include_statistics=False)
        
        # All should return valid results
        for result in [result1, result2, result3, result4]:
            assert isinstance(result, dict)
            assert "figure_type" in result


class TestBaseAnalyzerErrorConditions:
    """Test BaseAnalyzer error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = ConcreteAnalyzer()
    
    @pytest.mark.unit
    def test_analyze_with_none_figure(self):
        """Test analyze with None figure."""
        result = self.analyzer.analyze(None)
        
        assert isinstance(result, dict)
        assert result["figure_type"] == "test_figure"
        assert result["axes_count"] == 0
    
    @pytest.mark.unit
    def test_analyze_with_invalid_figure(self):
        """Test analyze with invalid figure object."""
        result = self.analyzer.analyze("not a figure")
        
        assert isinstance(result, dict)
        assert result["figure_type"] == "test_figure"
    
    @pytest.mark.unit
    def test_extract_basic_info_with_errors(self):
        """Test extract_basic_info when methods raise errors."""
        # Create an analyzer that raises errors
        class ErrorAnalyzer(ConcreteAnalyzer):
            def _get_title(self, figure):
                raise Exception("Title error")
        
        analyzer = ErrorAnalyzer()
        
        # Should raise the exception (base_analyzer doesn't handle errors internally)
        fig, ax = plt.subplots()
        with pytest.raises(Exception, match="Title error"):
            analyzer.extract_basic_info(fig)
        plt.close()


class TestBaseAnalyzerInheritance:
    """Test BaseAnalyzer inheritance patterns."""
    
    @pytest.mark.unit
    def test_minimal_concrete_implementation(self):
        """Test minimal concrete implementation."""
        class MinimalAnalyzer(BaseAnalyzer):
            def analyze(self, figure, **kwargs):
                return {"minimal": True}
            
            def _get_figure_type(self, figure):
                return "minimal"
            
            def _get_dimensions(self, figure):
                return (0, 0)
            
            def _get_title(self, figure):
                return None
            
            def _get_axes(self, figure):
                return []
            
            def _get_axes_count(self, figure):
                return 0
        
        analyzer = MinimalAnalyzer()
        result = analyzer.analyze("test")
        
        assert result == {"minimal": True}
    
    @pytest.mark.unit
    def test_method_override_patterns(self):
        """Test method override patterns."""
        class OverrideAnalyzer(ConcreteAnalyzer):
            def _get_data_points(self, figure):
                return 42
            
            def _has_grid(self, ax):
                return True
            
            def _get_colors(self, figure):
                return [{"hex": "#ff0000", "name": "red"}]
        
        analyzer = OverrideAnalyzer()
        
        assert analyzer._get_data_points("test") == 42
        assert analyzer._has_grid("test") is True
        colors = analyzer._get_colors("test")
        assert len(colors) == 1
        assert colors[0]["hex"] == "#ff0000"


if __name__ == "__main__":
    pytest.main([__file__]) 