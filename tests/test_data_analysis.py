#!/usr/bin/env python3
"""
Advanced tests for data analysis capabilities that provide rich information
for LLMs to understand and analyze the data presented in plots.
"""

import pytest
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot2llm import FigureConverter
from plot2llm.analyzers import MatplotlibAnalyzer


class TestDataAnalysisForLLMs:
    """Test cases focused on extracting data insights for LLM analysis."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MatplotlibAnalyzer()
        self.converter = FigureConverter()
        
    def teardown_method(self):
        """Clean up after tests."""
        plt.close('all')
    
    def test_trend_analysis_line_plot(self):
        """Test analysis of trends in line plots for LLM understanding."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create data with clear trends
        x = np.linspace(0, 20, 100)
        y_linear = 2 * x + 10 + np.random.normal(0, 2, 100)  # Linear trend with noise
        y_exponential = 5 * np.exp(0.1 * x) + np.random.normal(0, 10, 100)  # Exponential trend
        
        ax.plot(x, y_linear, 'b-', label='Linear Trend', linewidth=2)
        ax.plot(x, y_exponential, 'r--', label='Exponential Trend', linewidth=2)
        ax.set_title('Trend Analysis: Linear vs Exponential Growth')
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Analyze with high detail
        result = self.analyzer.analyze(fig, detail_level="high", include_statistics=True)
        
        # Test that we get meaningful trend information
        assert 'data_info' in result
        assert 'statistics' in result['data_info']
        
        # Check for trend indicators
        stats = result['data_info']['statistics']
        if 'global' in stats:
            assert 'mean' in stats['global']
        else:
            assert 'mean' in stats
        
        # Test conversion to text for LLM consumption
        text_result = self.converter.convert(fig, output_format='text')
        assert 'trend' in text_result.lower() or 'growth' in text_result.lower()
        assert 'linear' in text_result.lower()
        assert 'exponential' in text_result.lower()
    
    def test_correlation_analysis_scatter(self):
        """Test analysis of correlations in scatter plots."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Strong positive correlation
        np.random.seed(42)
        x1 = np.random.normal(0, 1, 100)
        y1 = 2 * x1 + np.random.normal(0, 0.5, 100)  # Strong positive correlation
        
        # Weak correlation
        x2 = np.random.normal(0, 1, 100)
        y2 = np.random.normal(0, 1, 100)  # Weak/no correlation
        
        ax1.scatter(x1, y1, alpha=0.6, color='blue')
        ax1.set_title('Strong Positive Correlation')
        ax1.set_xlabel('Variable X')
        ax1.set_ylabel('Variable Y')
        
        ax2.scatter(x2, y2, alpha=0.6, color='red')
        ax2.set_title('Weak/No Correlation')
        ax2.set_xlabel('Variable X')
        ax2.set_ylabel('Variable Y')
        
        result = self.analyzer.analyze(fig, detail_level="high")
        
        # Test that we get correlation-relevant information
        assert len(result['axes_info']) == 2
        assert result['axes_info'][0]['title'] == 'Strong Positive Correlation'
        assert result['axes_info'][1]['title'] == 'Weak/No Correlation'
        
        # Test text conversion for correlation analysis
        text_result = self.converter.convert(fig, output_format='text')
        assert 'correlation' in text_result.lower()
        assert 'scatter' in text_result.lower()
    
    def test_distribution_analysis_histogram(self):
        """Test analysis of data distributions in histograms."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Normal distribution
        normal_data = np.random.normal(0, 1, 1000)
        ax1.hist(normal_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Normal Distribution')
        ax1.set_xlabel('Values')
        ax1.set_ylabel('Frequency')
        
        # Skewed distribution
        skewed_data = np.random.exponential(2, 1000)
        ax2.hist(skewed_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.set_title('Skewed Distribution (Exponential)')
        ax2.set_xlabel('Values')
        ax2.set_ylabel('Frequency')
        
        # Bimodal distribution
        bimodal_data = np.concatenate([
            np.random.normal(-2, 0.5, 500),
            np.random.normal(2, 0.5, 500)
        ])
        ax3.hist(bimodal_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_title('Bimodal Distribution')
        ax3.set_xlabel('Values')
        ax3.set_ylabel('Frequency')
        
        result = self.analyzer.analyze(fig, detail_level="high")
        
        # Test distribution analysis
        assert len(result['axes_info']) == 3
        plot_types = result['data_info']['plot_types']
        if isinstance(plot_types, list):
            assert any(pt.get('type') == 'histogram' for pt in plot_types)
        else:
            assert 'histogram' in plot_types
        
        # Test text conversion for distribution analysis
        text_result = self.converter.convert(fig, output_format='text')
        assert 'distribution' in text_result.lower()
        assert 'histogram' in text_result.lower()
        assert 'frequency' in text_result.lower()
    
    def test_comparative_analysis_bar_chart(self):
        """Test analysis of comparative data in bar charts."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Simple bar chart
        categories = ['Category A', 'Category B', 'Category C', 'Category D']
        values = [23, 45, 56, 78]
        colors = ['red', 'blue', 'green', 'orange']
        
        bars1 = ax1.bar(categories, values, color=colors)
        ax1.set_title('Simple Bar Chart')
        ax1.set_xlabel('Categories')
        ax1.set_ylabel('Values')
        
        # Add value labels
        for bar, value in zip(bars1, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{value}', ha='center', va='bottom')
        
        # Grouped bar chart
        x = np.arange(len(categories))
        width = 0.35
        
        group1_values = [20, 35, 30, 35]
        group2_values = [25, 32, 34, 20]
        
        bars2_1 = ax2.bar(x - width/2, group1_values, width, label='Group 1')
        bars2_2 = ax2.bar(x + width/2, group2_values, width, label='Group 2')
        
        ax2.set_title('Grouped Bar Chart')
        ax2.set_xlabel('Categories')
        ax2.set_ylabel('Values')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        
        result = self.analyzer.analyze(fig, detail_level="high")
        
        # Test comparative analysis
        assert len(result['axes_info']) == 2
        plot_types = result['data_info']['plot_types']
        if isinstance(plot_types, list):
            assert any(pt.get('type') == 'bar' for pt in plot_types)
        else:
            assert 'bar' in plot_types
        
        # Test text conversion for comparative analysis
        text_result = self.converter.convert(fig, output_format='text')
        assert 'bar' in text_result.lower()
        assert 'category' in text_result.lower()
        assert 'group' in text_result.lower()
    
    def test_outlier_detection(self):
        """Test detection and analysis of outliers in data."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create data with outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        outliers = np.array([5, -4, 6, -5])  # Clear outliers
        
        all_data = np.concatenate([normal_data, outliers])
        
        ax.scatter(range(len(all_data)), all_data, alpha=0.6, color='blue')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Mean')
        ax.set_title('Data with Outliers')
        ax.set_xlabel('Data Point Index')
        ax.set_ylabel('Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        result = self.analyzer.analyze(fig, detail_level="high")
        
        # Test outlier analysis
        assert 'data_info' in result
        assert 'statistics' in result['data_info']
        
        stats = result['data_info']['statistics']
        if 'global' in stats:
            assert 'min' in stats['global']
            assert 'max' in stats['global']
        else:
            assert 'min' in stats
            assert 'max' in stats
        
        # Test text conversion for outlier analysis
        text_result = self.converter.convert(fig, output_format='text')
        assert 'outlier' in text_result.lower() or 'extreme' in text_result.lower()
    
    def test_time_series_analysis(self):
        """Test analysis of time series data patterns."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Create time series data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Seasonal pattern
        seasonal_data = 10 + 5 * np.sin(2 * np.pi * np.arange(100) / 30) + np.random.normal(0, 1, 100)
        ax1.plot(dates, seasonal_data, 'b-', linewidth=2)
        ax1.set_title('Time Series with Seasonal Pattern')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Values')
        ax1.grid(True, alpha=0.3)
        
        # Trend with noise
        trend_data = 5 + 0.1 * np.arange(100) + np.random.normal(0, 2, 100)
        ax2.plot(dates, trend_data, 'r-', linewidth=2)
        ax2.set_title('Time Series with Upward Trend')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Values')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        result = self.analyzer.analyze(fig, detail_level="high")
        
        # Test time series analysis
        assert len(result['axes_info']) == 2
        plot_types = result['data_info']['plot_types']
        if isinstance(plot_types, list):
            assert any(pt.get('type') == 'line' for pt in plot_types)
        else:
            assert 'line' in plot_types
        
        # Test text conversion for time series analysis
        text_result = self.converter.convert(fig, output_format='text')
        assert 'time' in text_result.lower() or 'date' in text_result.lower()
        assert 'trend' in text_result.lower() or 'pattern' in text_result.lower()
    
    def test_statistical_summary_for_llm(self):
        """Test comprehensive statistical summary for LLM analysis."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create complex dataset
        np.random.seed(42)
        data1 = np.random.normal(10, 2, 200)
        data2 = np.random.normal(15, 3, 200)
        data3 = np.random.normal(8, 1.5, 200)
        
        # Create box plot for statistical comparison
        box_data = [data1, data2, data3]
        labels = ['Group A', 'Group B', 'Group C']
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title('Statistical Comparison: Box Plots')
        ax.set_xlabel('Groups')
        ax.set_ylabel('Values')
        ax.grid(True, alpha=0.3)
        
        result = self.analyzer.analyze(fig, detail_level="high")
        
        # Test statistical summary
        assert 'data_info' in result
        assert 'statistics' in result['data_info']
        
        # Test text conversion for statistical analysis
        text_result = self.converter.convert(fig, output_format='text')
        assert 'statistical' in text_result.lower() or 'comparison' in text_result.lower()
        assert 'box' in text_result.lower()
        assert 'group' in text_result.lower()
    
    def test_data_quality_indicators(self):
        """Test analysis of data quality indicators."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Clean data
        clean_data = np.random.normal(0, 1, 100)
        ax1.scatter(range(len(clean_data)), clean_data, alpha=0.6, color='green')
        ax1.set_title('Clean Data (No Missing Values)')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Values')
        
        # Data with missing values (NaN)
        data_with_nans = np.random.normal(0, 1, 100)
        data_with_nans[::10] = np.nan  # Every 10th value is NaN
        valid_indices = ~np.isnan(data_with_nans)
        
        ax2.scatter(np.arange(len(data_with_nans))[valid_indices], 
                   data_with_nans[valid_indices], alpha=0.6, color='red')
        ax2.set_title('Data with Missing Values (NaN)')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Values')
        
        result = self.analyzer.analyze(fig, detail_level="high")
        
        # Test data quality analysis
        assert len(result['axes_info']) == 2
        plot_types = result['data_info']['plot_types']
        if isinstance(plot_types, list):
            assert any(pt.get('type') == 'scatter' for pt in plot_types)
        else:
            assert 'scatter' in plot_types
        
        # Test text conversion for data quality
        text_result = self.converter.convert(fig, output_format='text')
        assert 'data' in text_result.lower()
        assert 'missing' in text_result.lower() or 'clean' in text_result.lower()
    
    def test_semantic_analysis_output(self):
        """Test semantic analysis output for LLM consumption."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a meaningful dataset
        years = np.arange(2010, 2023)
        sales = [100, 120, 150, 180, 220, 280, 350, 420, 500, 600, 720, 850, 1000]
        
        ax.plot(years, sales, 'bo-', linewidth=2, markersize=8)
        ax.set_title('Company Sales Growth (2010-2022)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Sales (in thousands)')
        ax.grid(True, alpha=0.3)
        
        # Test semantic conversion
        semantic_result = self.converter.convert(fig, output_format='semantic')
        
        # Test that semantic output contains meaningful structure
        assert isinstance(semantic_result, str)
        
        # Test text conversion for business insights
        text_result = self.converter.convert(fig, output_format='text')
        assert 'sales' in text_result.lower()
        assert 'growth' in text_result.lower()
        assert 'year' in text_result.lower() 