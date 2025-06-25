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
        assert 'Test Plot' in result
        assert 'line' in result.lower()
        assert 'Data Series' in result
    
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
        
        assert 'Multi-plot Figure' in result
        assert 'Plot 1' in result
        assert 'Plot 2' in result
        assert 'scatter' in result.lower()
        assert 'bar' in result.lower()
    
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
        
        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed['figure_type'] == 'matplotlib'
        assert parsed['title'] == 'Test Plot'
        assert len(parsed['axes']) == 1
    
    def test_format_complex_data(self):
        """Test formatting complex plot data to JSON."""
        plot_data = {
            'figure_type': 'matplotlib',
            'title': 'Complex Plot',
            'axes': [
                {
                    'title': 'Scatter Plot',
                    'xlabel': 'X',
                    'ylabel': 'Y',
                    'plot_types': [
                        {'type': 'scatter', 'label': 'Points'},
                        {'type': 'line', 'label': 'Trend'}
                    ]
                }
            ]
        }
        
        result = self.formatter.format(plot_data)
        parsed = json.loads(result)
        
        assert parsed['title'] == 'Complex Plot'
        assert len(parsed['axes'][0]['plot_types']) == 2
        assert parsed['axes'][0]['plot_types'][0]['type'] == 'scatter'
        assert parsed['axes'][0]['plot_types'][1]['type'] == 'line'
    
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
        # Should contain semantic descriptions
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
        assert 'Data Points' in result
        assert 'Trend Line' in result
    
    def test_format_invalid_input(self):
        """Test error handling for invalid input."""
        with pytest.raises(ValueError, match="Invalid plot data"):
            self.formatter.format(None) 