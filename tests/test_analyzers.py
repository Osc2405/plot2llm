"""
Tests for the analyzer classes.
"""

import pytest
import matplotlib.pyplot as plt
import numpy as np
from python2llm.analyzers import BaseAnalyzer, MatplotlibAnalyzer


class TestBaseAnalyzer:
    """Test cases for the base BaseAnalyzer class."""
    
    def test_base_analyzer_initialization(self):
        """Test that base analyzer initializes correctly."""
        analyzer = BaseAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze')
    
    def test_base_analyzer_analyze_raises_not_implemented(self):
        """Test that base analyzer raises NotImplementedError."""
        analyzer = BaseAnalyzer()
        fig, ax = plt.subplots()
        
        with pytest.raises(NotImplementedError):
            analyzer.analyze(fig)
        
        plt.close(fig)


class TestMatplotlibAnalyzer:
    """Test cases for MatplotlibAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MatplotlibAnalyzer()
        
    def teardown_method(self):
        """Clean up after tests."""
        plt.close('all')
    
    def test_analyzer_initialization(self):
        """Test that matplotlib analyzer initializes correctly."""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'analyze')
    
    def test_analyze_simple_line_plot(self):
        """Test analyzing a simple line plot."""
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y, label="Sine Wave")
        ax.set_title("Test Plot")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.legend()
        
        result = self.analyzer.analyze(fig)
        
        assert isinstance(result, dict)
        assert 'figure_type' in result
        assert 'axes' in result
        assert 'title' in result
        assert result['figure_type'] == 'matplotlib'
        assert result['title'] == "Test Plot"
        assert len(result['axes']) == 1
    
    def test_analyze_multiple_axes(self):
        """Test analyzing a figure with multiple axes."""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        # First subplot
        ax1.plot([1, 2, 3], [1, 4, 2])
        ax1.set_title("Plot 1")
        
        # Second subplot
        ax2.scatter([1, 2, 3], [1, 4, 2])
        ax2.set_title("Plot 2")
        
        result = self.analyzer.analyze(fig)
        
        assert len(result['axes']) == 2
        assert result['axes'][0]['title'] == "Plot 1"
        assert result['axes'][1]['title'] == "Plot 2"
    
    def test_analyze_scatter_plot(self):
        """Test analyzing a scatter plot."""
        fig, ax = plt.subplots()
        x = np.random.randn(50)
        y = np.random.randn(50)
        ax.scatter(x, y, alpha=0.6)
        ax.set_title("Scatter Plot")
        
        result = self.analyzer.analyze(fig)
        
        assert result['axes'][0]['plot_types'][0]['type'] == 'scatter'
    
    def test_analyze_bar_plot(self):
        """Test analyzing a bar plot."""
        fig, ax = plt.subplots()
        categories = ['A', 'B', 'C', 'D']
        values = [4, 3, 2, 1]
        ax.bar(categories, values)
        ax.set_title("Bar Chart")
        
        result = self.analyzer.analyze(fig)
        
        assert result['axes'][0]['plot_types'][0]['type'] == 'bar'
    
    def test_analyze_histogram(self):
        """Test analyzing a histogram."""
        fig, ax = plt.subplots()
        data = np.random.randn(1000)
        ax.hist(data, bins=30)
        ax.set_title("Histogram")
        
        result = self.analyzer.analyze(fig)
        
        assert result['axes'][0]['plot_types'][0]['type'] == 'histogram'
    
    def test_analyze_empty_figure(self):
        """Test analyzing an empty figure."""
        fig, ax = plt.subplots()
        
        result = self.analyzer.analyze(fig)
        
        assert result['figure_type'] == 'matplotlib'
        assert len(result['axes']) == 1
        assert len(result['axes'][0]['plot_types']) == 0
    
    def test_analyze_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="Invalid figure object"):
            self.analyzer.analyze(None)
        
        with pytest.raises(ValueError, match="Not a matplotlib figure"):
            self.analyzer.analyze("not a figure") 