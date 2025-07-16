#!/usr/bin/env python3
"""
Tests for the formatters module.
"""

import pytest
import json
from plot2llm.formatters import TextFormatter, JSONFormatter, SemanticFormatter


class TestTextFormatter:
    """Test cases for TextFormatter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = TextFormatter()
    
    def test_formatter_initialization(self):
        """Test that text formatter initializes correctly."""
        assert self.formatter is not None
        assert hasattr(self.formatter, 'format')
    
    def test_format_simple_plot_data(self):
        """Test formatting simple plot data."""
        plot_data = {
            'figure_type': 'matplotlib',
            'title': 'Test Plot',
            'axes': [{
                'title': 'Subplot 1',
                'xlabel': 'X Axis',
                'ylabel': 'Y Axis',
                'plot_types': [{
                    'type': 'line',
                    'label': 'Data Series'
                }]
            }]
        }
        
        result = self.formatter.format(plot_data)
        
        assert isinstance(result, str)
        assert 'Test Plot' in result or 'test plot' in result.lower()
        assert 'line' in result.lower()
        assert 'Data Series' in result or 'data series' in result
    
    def test_format_multiple_axes(self):
        """Test formatting data with multiple axes."""
        plot_data = {
            'figure_type': 'matplotlib',
            'title': 'Multi-plot Figure',
            'axes': [
                {
                    'title': 'Plot 1',
                    'plot_types': [{'type': 'scatter'}]
                },
                {
                    'title': 'Plot 2',
                    'plot_types': [{'type': 'bar'}]
                }
            ]
        }
        
        result = self.formatter.format(plot_data)
        
        assert 'Multi-plot Figure' in result or 'multi-plot figure' in result.lower()
        assert 'Plot 1' in result or 'plot 1' in result.lower()
        assert 'Plot 2' in result or 'plot 2' in result.lower()
        assert 'scatter' in result.lower()
        assert 'bar' in result.lower()
    
    def test_format_with_axis_types(self):
        """Test formatting data with explicit axis types."""
        plot_data = {
            'figure_type': 'matplotlib',
            'title': 'Sales Data',
            'axes': [{
                'title': 'Monthly Sales',
                'xlabel': 'Month',
                'ylabel': 'Sales',
                'x_type': 'CATEGORY',
                'y_type': 'NUMERIC',
                'plot_types': [{'type': 'bar'}]
            }]
        }
        
        result = self.formatter.format(plot_data)
        
        assert 'type: CATEGORY' in result
        assert 'type: NUMERIC' in result
        assert 'X-axis: Month' in result
        assert 'Y-axis: Sales' in result
    
    def test_format_with_curve_points(self):
        """Test formatting data with curve points."""
        plot_data = {
            'figure_type': 'matplotlib',
            'title': 'Point Data',
            'axes': [{
                'title': 'Points Plot',
                'x_type': 'CATEGORY',
                'y_type': 'NUMERIC',
                'curve_points': [
                    {'x': ['Jan'], 'y': 10, 'label': 'Series A'},
                    {'x': ['Feb'], 'y': 20, 'label': 'Series A'},
                    {'x': ['Mar'], 'y': 30, 'label': 'Series A'},
                ]
            }]
        }
        
        result = self.formatter.format(plot_data)
        
        assert 'Curve points:' in result
        assert 'Point 1:' in result
        assert '[Series A]' in result
        assert 'categories: ' in result
        assert 'Jan' in result and 'Feb' in result
    
    def test_format_with_date_axis(self):
        """Test formatting data with date axis type."""
        plot_data = {
            'figure_type': 'matplotlib',
            'title': 'Time Series',
            'axes': [{
                'title': 'Daily Data',
                'x_type': 'DATE',
                'y_type': 'NUMERIC',
                'curve_points': [
                    {'x': '2023-01-01', 'y': 100},
                    {'x': '2023-01-02', 'y': 150},
                ]
            }]
        }
        
        result = self.formatter.format(plot_data)
        
        assert 'type: DATE' in result
        assert 'date: ' in result
        assert '2023-01-01' in result
    
    def test_format_empty_data(self):
        """Test formatting empty plot data."""
        plot_data = {
            'figure_type': 'matplotlib',
            'title': '',
            'axes': []
        }
        
        result = self.formatter.format(plot_data)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_format_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="Invalid plot data"):
            self.formatter.format(None)
        
        with pytest.raises(ValueError, match="Invalid plot data"):
            self.formatter.format("not a dict")


class TestJSONFormatter:
    """Test cases for JSONFormatter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = JSONFormatter()
    
    def test_formatter_initialization(self):
        """Test that JSON formatter initializes correctly."""
        assert self.formatter is not None
        assert hasattr(self.formatter, 'format')
    
    def test_format_simple_plot_data(self):
        """Test formatting simple plot data to JSON."""
        plot_data = {
            'figure_type': 'matplotlib',
            'title': 'Test Plot',
            'axes': [{
                'title': 'Subplot 1',
                'plot_types': [{'type': 'line'}]
            }]
        }
        
        result = self.formatter.format(plot_data)
        
        assert isinstance(result, dict)
        assert result['figure_type'] == 'matplotlib'
        assert result['title'] == 'Test Plot'
        assert len(result['axes']) == 1
    
    def test_format_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="Invalid plot data"):
            self.formatter.format(None)


class TestSemanticFormatter:
    """Test cases for SemanticFormatter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = SemanticFormatter()
    
    def test_formatter_initialization(self):
        """Test that semantic formatter initializes correctly."""
        assert self.formatter is not None
        assert hasattr(self.formatter, 'format')
    
    def test_format_simple_plot_data(self):
        """Test formatting simple plot data to semantic format."""
        plot_data = {
            'figure_type': 'matplotlib',
            'title': 'Test Plot',
            'axes': [{
                'title': 'Subplot 1',
                'plot_types': [{'type': 'line'}]
            }]
        }
        
        result = self.formatter.format(plot_data)
        
        assert isinstance(result, str)
        assert 'visualization' in result.lower() or 'chart' in result.lower()
        assert 'line' in result.lower()
    
    def test_format_multiple_plot_types(self):
        """Test formatting data with multiple plot types."""
        plot_data = {
            'figure_type': 'matplotlib',
            'title': 'Mixed Chart',
            'axes': [{
                'title': 'Mixed Plot',
                'plot_types': [
                    {'type': 'scatter', 'label': 'Data Points'},
                    {'type': 'line', 'label': 'Trend Line'}
                ]
            }]
        }
        
        result = self.formatter.format(plot_data)
        
        assert 'scatter' in result.lower()
        assert 'line' in result.lower()
        assert 'Data Points' in result or 'data points' in result
    
    def test_format_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="Invalid plot data"):
            self.formatter.format(None) 