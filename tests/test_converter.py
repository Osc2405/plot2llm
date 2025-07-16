#!/usr/bin/env python3
"""
Tests for the converter module.
"""

import pytest
import matplotlib.pyplot as plt
import numpy as np
from plot2llm import FigureConverter
from plot2llm.analyzers import MatplotlibAnalyzer
from plot2llm.formatters import TextFormatter, JSONFormatter


class TestFigureConverter:
    """Test cases for FigureConverter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = FigureConverter()
        
    def teardown_method(self):
        """Clean up after tests."""
        plt.close('all')
    
    def test_converter_initialization(self):
        """Test that converter initializes correctly."""
        assert self.converter is not None
        assert hasattr(self.converter, 'analyzers')
        assert hasattr(self.converter, 'formatters')
        assert hasattr(self.converter, 'register_analyzer')
        assert hasattr(self.converter, 'register_formatter')
    
    def test_register_analyzer(self):
        """Test registering a custom analyzer."""
        analyzer = MatplotlibAnalyzer()
        self.converter.register_analyzer('custom_matplotlib', analyzer)
        assert 'custom_matplotlib' in self.converter.analyzers
    
    def test_register_formatter(self):
        """Test registering a custom formatter."""
        formatter = TextFormatter()
        self.converter.register_formatter('custom_text', formatter)
        assert 'custom_text' in self.converter.formatters
    
    def test_convert_simple_plot(self):
        """Test converting a simple matplotlib plot."""
        # Create a simple plot
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_title("Simple Sine Wave")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Convert to text
        result = self.converter.convert(fig, output_format='text')
        assert isinstance(result, str)
        assert len(result) > 0
        assert "sine" in result.lower() or "wave" in result.lower()
    
    def test_convert_to_json(self):
        """Test converting a plot to JSON format."""
        # Create a simple plot
        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        y = np.cos(x)
        ax.plot(x, y, label="Cosine")
        ax.legend()
        
        # Convert to JSON
        result = self.converter.convert(fig, output_format='json')
        assert isinstance(result, str)
        # Should be valid JSON
        import json
        json.loads(result)
    
    def test_convert_unsupported_format(self):
        """Test error handling for unsupported output format."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        with pytest.raises(ValueError, match="Unsupported output format"):
            self.converter.convert(fig, output_format='unsupported')
    
    def test_convert_invalid_figure(self):
        """Test error handling for invalid figure object."""
        with pytest.raises(ValueError, match="Invalid figure object"):
            self.converter.convert(None, output_format='text')
    
    def test_auto_detect_figure_type(self):
        """Test automatic figure type detection."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        # Should automatically detect matplotlib figure
        result = self.converter.convert(fig, output_format='text')
        assert isinstance(result, str)
        assert len(result) > 0 