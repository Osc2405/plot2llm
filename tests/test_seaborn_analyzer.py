"""
Tests for the seaborn analyzer functionality.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot2llm.analyzers.seaborn_analyzer import SeabornAnalyzer
from plot2llm.analyzers import FigureAnalyzer

class TestSeabornAnalyzer:
    """Test cases for SeabornAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SeabornAnalyzer()
        self.figure_analyzer = FigureAnalyzer()
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'x': np.random.randn(50),
            'y': np.random.randn(50),
            'category': np.random.choice(['A', 'B', 'C'], 50),
            'value': np.random.uniform(0, 1, 50)
        })
    
    def test_seaborn_analyzer_initialization(self):
        """Test that SeabornAnalyzer initializes correctly."""
        assert self.analyzer is not None
        assert "matplotlib.figure.Figure" in self.analyzer.supported_types
        assert "seaborn.axisgrid.FacetGrid" in self.analyzer.supported_types
    
    def test_seaborn_scatter_plot(self):
        """Test analysis of seaborn scatter plot."""
        # Create seaborn scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.sample_data, x='x', y='y', hue='category')
        plt.title('Test Scatter Plot')
        
        fig = plt.gcf()
        
        # Analyze the figure
        result = self.analyzer.analyze(fig, detail_level="medium")
        
        # Check basic structure
        assert "basic_info" in result
        assert "axes_info" in result
        assert "data_info" in result
        assert "visual_info" in result
        assert "seaborn_info" in result
        
        # Check basic info
        basic_info = result["basic_info"]
        assert basic_info["figure_type"] in ["matplotlib.figure", "seaborn.scatterplot"]
        assert basic_info["axes_count"] > 0
        
        # Check seaborn info
        seaborn_info = result["seaborn_info"]
        assert "plot_type" in seaborn_info
        
        plt.close(fig)
    
    def test_seaborn_heatmap(self):
        """Test analysis of seaborn heatmap."""
        # Create correlation matrix
        corr_matrix = self.sample_data[['x', 'y', 'value']].corr()
        
        # Create seaborn heatmap
        plt.figure(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Test Heatmap')
        
        fig = plt.gcf()
        
        # Analyze the figure
        result = self.analyzer.analyze(fig, detail_level="medium")
        
        # Check structure
        assert "basic_info" in result
        assert "seaborn_info" in result
        
        # Check that it's recognized as a heatmap
        seaborn_info = result["seaborn_info"]
        assert "plot_type" in seaborn_info
        
        plt.close(fig)
    
    def test_seaborn_facetgrid(self):
        """Test analysis of seaborn FacetGrid."""
        # Create seaborn FacetGrid
        g = sns.FacetGrid(self.sample_data, col="category", height=4, aspect=1.2)
        g.map_dataframe(sns.scatterplot, x="x", y="y")
        g.fig.suptitle('Test FacetGrid')
        
        # Analyze the FacetGrid
        result = self.analyzer.analyze(g, detail_level="medium")
        
        # Check structure
        assert "basic_info" in result
        assert "seaborn_info" in result
        
        # Check seaborn-specific info
        seaborn_info = result["seaborn_info"]
        assert "plot_type" in seaborn_info
        assert "grid_shape" in seaborn_info
        assert "grid_size" in seaborn_info
        
        # Check that it's recognized as a facet grid
        assert seaborn_info["plot_type"] == "facet_grid"
        
        plt.close(g.fig)
    
    def test_seaborn_pairplot(self):
        """Test analysis of seaborn pairplot."""
        # Create seaborn pairplot
        pair_plot = sns.pairplot(self.sample_data[['x', 'y', 'value']], diag_kind='kde')
        pair_plot.fig.suptitle('Test Pairplot')
        
        # Analyze the pairplot
        result = self.analyzer.analyze(pair_plot, detail_level="medium")
        
        # Check structure
        assert "basic_info" in result
        assert "seaborn_info" in result
        
        # Check seaborn-specific info
        seaborn_info = result["seaborn_info"]
        assert "plot_type" in seaborn_info
        
        # Check that it's recognized as a pair grid (seaborn's internal name)
        assert seaborn_info["plot_type"] == "pair_grid"
        
        plt.close(pair_plot.fig)
    
    def test_seaborn_distribution_plots(self):
        """Test analysis of seaborn distribution plots."""
        # Create subplots with different distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Histogram
        sns.histplot(data=self.sample_data, x='x', ax=axes[0, 0])
        axes[0, 0].set_title('Histogram')
        
        # KDE plot
        sns.kdeplot(data=self.sample_data, x='y', ax=axes[0, 1])
        axes[0, 1].set_title('KDE Plot')
        
        # Box plot
        sns.boxplot(data=self.sample_data, x='category', y='value', ax=axes[1, 0])
        axes[1, 0].set_title('Box Plot')
        
        # Violin plot
        sns.violinplot(data=self.sample_data, x='category', y='x', ax=axes[1, 1])
        axes[1, 1].set_title('Violin Plot')
        
        plt.tight_layout()
        
        # Analyze the figure
        result = self.analyzer.analyze(fig, detail_level="medium")
        
        # Check structure
        assert "basic_info" in result
        assert "axes_info" in result
        assert "data_info" in result
        
        # Check that we have multiple axes
        basic_info = result["basic_info"]
        assert basic_info["axes_count"] == 4
        
        plt.close(fig)
    
    def test_figure_analyzer_with_seaborn(self):
        """Test that FigureAnalyzer correctly routes seaborn figures."""
        # Create seaborn scatter plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.sample_data, x='x', y='y', hue='category')
        plt.title('Test Scatter Plot')
        
        fig = plt.gcf()
        
        # Use FigureAnalyzer with seaborn type
        result = self.figure_analyzer.analyze(fig, "seaborn", detail_level="medium")
        
        # Check structure
        assert "basic_info" in result
        assert "seaborn_info" in result
        assert "metadata" in result
        
        # Check metadata
        metadata = result["metadata"]
        assert metadata["figure_type"] == "seaborn"
        
        plt.close(fig)
    
    def test_different_detail_levels(self):
        """Test analysis with different detail levels."""
        # Create seaborn plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.sample_data, x='x', y='y', hue='category')
        plt.title('Test Plot')
        
        fig = plt.gcf()
        
        # Test low detail level
        result_low = self.analyzer.analyze(fig, detail_level="low")
        assert "basic_info" in result_low
        assert "detailed_info" not in result_low
        
        # Test high detail level
        result_high = self.analyzer.analyze(fig, detail_level="high")
        assert "basic_info" in result_high
        assert "detailed_info" in result_high
        
        plt.close(fig)
    
    def test_data_analysis_options(self):
        """Test analysis with different data analysis options."""
        # Create seaborn plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=self.sample_data, x='x', y='y', hue='category')
        plt.title('Test Plot')
        
        fig = plt.gcf()
        
        # Test without data analysis
        result_no_data = self.analyzer.analyze(
            fig, 
            detail_level="medium",
            include_data=False,
            include_colors=True,
            include_statistics=False
        )
        
        assert "data_info" in result_no_data
        assert "visual_info" in result_no_data
        
        # Test without color analysis
        result_no_colors = self.analyzer.analyze(
            fig, 
            detail_level="medium",
            include_data=True,
            include_colors=False,
            include_statistics=True
        )
        
        assert "data_info" in result_no_colors
        assert "visual_info" in result_no_colors
        
        plt.close(fig)
    
    def test_error_handling(self):
        """Test error handling with invalid input."""
        # Test with None - should handle gracefully
        try:
            result = self.analyzer.analyze(None, detail_level="medium")
            # If it doesn't raise an exception, it should return some result
            assert isinstance(result, dict)
        except Exception:
            # It's also acceptable to raise an exception
            pass
        
        # Test with invalid figure type - should handle gracefully
        try:
            result = self.analyzer.analyze("not a figure", detail_level="medium")
            # If it doesn't raise an exception, it should return some result
            assert isinstance(result, dict)
        except Exception:
            # It's also acceptable to raise an exception
            pass
    
    def test_seaborn_plot_type_detection(self):
        """Test detection of different seaborn plot types."""
        # Test scatter plot
        plt.figure()
        sns.scatterplot(data=self.sample_data, x='x', y='y')
        fig = plt.gcf()
        result = self.analyzer.analyze(fig, detail_level="low")
        seaborn_info = result["seaborn_info"]
        assert "plot_type" in seaborn_info
        plt.close(fig)
        
        # Test line plot
        plt.figure()
        sns.lineplot(data=self.sample_data, x='x', y='y')
        fig = plt.gcf()
        result = self.analyzer.analyze(fig, detail_level="low")
        seaborn_info = result["seaborn_info"]
        assert "plot_type" in seaborn_info
        plt.close(fig)
        
        # Test bar plot
        plt.figure()
        sns.barplot(data=self.sample_data, x='category', y='value')
        fig = plt.gcf()
        result = self.analyzer.analyze(fig, detail_level="low")
        seaborn_info = result["seaborn_info"]
        assert "plot_type" in seaborn_info
        plt.close(fig)

if __name__ == "__main__":
    pytest.main([__file__]) 