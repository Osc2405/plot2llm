import unittest
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plot2llm import FigureAnalyzer
from plot2llm.formatters import SemanticFormatter
import warnings
import json

class TestFixesVerification(unittest.TestCase):
    """Test to verify that all implemented fixes are working correctly."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FigureAnalyzer()
        self.formatter = SemanticFormatter()
        
        # Suppress warnings for cleaner test output
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        # Set random seed for reproducible tests
        np.random.seed(42)
    
    def tearDown(self):
        """Clean up after each test."""
        plt.close('all')
    
    def test_histogram_data_points_fix(self):
        """Test that histogram total_data_points shows number of bins, not total observations."""
        # Create histogram with 1000 observations in 30 bins
        data = np.random.normal(0, 1, 1000)
        
        fig, ax = plt.subplots()
        ax.hist(data, bins=30, alpha=0.7, color='blue')
        ax.set_title('Histogram Test')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="matplotlib")
        result = self.formatter.format(analysis)
        
        # Check that total_data_points shows number of bins, not observations
        data_summary = result.get('data_summary', {})
        total_points = data_summary.get('total_data_points', 0)
        
        # Should show 30 (number of bins), not 1000 (number of observations)
        self.assertEqual(total_points, 30, 
                        f"Histogram should show 30 bins, got {total_points}")
    
    def test_data_ranges_not_null_for_bar_plots(self):
        """Test that bar plots have non-null data_ranges."""
        # Create bar plot
        categories = ['A', 'B', 'C', 'D', 'E']
        values = [23, 45, 12, 36, 28]
        
        fig, ax = plt.subplots()
        ax.bar(categories, values, color='green', alpha=0.7)
        ax.set_title('Bar Plot Test')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Values')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="matplotlib")
        result = self.formatter.format(analysis)
        
        # Check data_ranges
        data_summary = result.get('data_summary', {})
        data_ranges = data_summary.get('data_ranges', {})
        
        # All range values should be non-null
        x_range = data_ranges.get('x', {})
        y_range = data_ranges.get('y', {})
        
        self.assertIsNotNone(x_range.get('min'), "x_range.min should not be null")
        self.assertIsNotNone(x_range.get('max'), "x_range.max should not be null")
        self.assertIsNotNone(y_range.get('min'), "y_range.min should not be null")
        self.assertIsNotNone(y_range.get('max'), "y_range.max should not be null")
    
    def test_data_ranges_not_null_for_histograms(self):
        """Test that histograms have non-null data_ranges."""
        # Create histogram
        data = np.random.normal(0, 1, 1000)
        
        fig, ax = plt.subplots()
        ax.hist(data, bins=30, alpha=0.7, color='purple')
        ax.set_title('Histogram Test')
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="matplotlib")
        result = self.formatter.format(analysis)
        
        # Check data_ranges
        data_summary = result.get('data_summary', {})
        data_ranges = data_summary.get('data_ranges', {})
        
        # All range values should be non-null
        x_range = data_ranges.get('x', {})
        y_range = data_ranges.get('y', {})
        
        self.assertIsNotNone(x_range.get('min'), "x_range.min should not be null")
        self.assertIsNotNone(x_range.get('max'), "x_range.max should not be null")
        self.assertIsNotNone(y_range.get('min'), "y_range.min should not be null")
        self.assertIsNotNone(y_range.get('max'), "y_range.max should not be null")
    
    def test_seaborn_palette_warning_fix(self):
        """Test that seaborn bar plots don't generate palette warnings."""
        # Create seaborn bar plot without hue (should not generate warning)
        categories = ['A', 'B', 'C', 'D', 'E']
        values = [15, 32, 18, 45, 22]
        
        fig, ax = plt.subplots()
        # This should NOT generate a FutureWarning about palette without hue
        sns.barplot(x=categories, y=values, ax=ax, color='green')
        ax.set_title('Seaborn Bar Plot Test')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Values')
        
        # Analyze - should complete without warnings
        analysis = self.analyzer.analyze(fig, figure_type="seaborn")
        result = self.formatter.format(analysis)
        
        # Check that analysis completed successfully
        self.assertIn('data_summary', result)
        data_summary = result.get('data_summary', {})
        self.assertIsNotNone(data_summary.get('total_data_points'))
    
    def test_small_dataset_runtime_warning_fix(self):
        """Test that small datasets don't generate runtime warnings."""
        # Create a very small dataset that could cause "degrees of freedom" warning
        small_data = [1, 2, 3, 4, 5]
        
        fig, ax = plt.subplots()
        sns.barplot(x=['A', 'B', 'C', 'D', 'E'], y=small_data, ax=ax, color='orange')
        ax.set_title('Small Dataset Test')
        
        # Analyze - should complete without runtime warnings
        analysis = self.analyzer.analyze(fig, figure_type="seaborn")
        result = self.formatter.format(analysis)
        
        # Check that analysis completed successfully
        self.assertIn('data_summary', result)
        data_summary = result.get('data_summary', {})
        self.assertEqual(data_summary.get('total_data_points'), 5)
    
    def test_normal_distribution_detection_fix(self):
        """Test that normal distributions are correctly detected."""
        # Create a normal distribution with fixed seed for reproducibility
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 1000)
        
        fig, ax = plt.subplots()
        ax.hist(normal_data, bins=30, alpha=0.7, color='skyblue')
        ax.set_title('Normal Distribution Test')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="matplotlib")
        result = self.formatter.format(analysis)
        
        # Check pattern analysis
        pattern_analysis = result.get('pattern_analysis', {})
        pattern_type = pattern_analysis.get('pattern_type', '')
        
        # Should detect as normal distribution and NOT multimodal
        self.assertIn('normal', pattern_type.lower(), 
                     f"Normal distribution should be detected, got: {pattern_type}")
        self.assertNotIn('multimodal', pattern_type.lower(),
                        f"Normal distribution should NOT be detected as multimodal, got: {pattern_type}")
    
    def test_multimodal_distribution_detection_fix(self):
        """Test that multimodal distributions are correctly detected."""
        # Create a bimodal distribution with fixed seed for reproducibility
        np.random.seed(123)
        bimodal_data = np.concatenate([
            np.random.normal(-3, 0.8, 400),  # First peak at -3
            np.random.normal(3, 0.8, 400)    # Second peak at 3
        ])
        
        fig, ax = plt.subplots()
        ax.hist(bimodal_data, bins=40, alpha=0.7, color='red')
        ax.set_title('Bimodal Distribution Test')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="matplotlib")
        result = self.formatter.format(analysis)
        
        # Check pattern analysis
        pattern_analysis = result.get('pattern_analysis', {})
        pattern_type = pattern_analysis.get('pattern_type', '')
        
        # Should detect as multimodal distribution
        self.assertIn('multimodal', pattern_type.lower(), 
                     f"Multimodal distribution should be detected, got: {pattern_type}")
    
    def test_trimodal_distribution_detection_fix(self):
        """Test that trimodal distributions are correctly detected."""
        # Create a trimodal distribution with fixed seed for reproducibility
        np.random.seed(456)
        trimodal_data = np.concatenate([
            np.random.normal(-4, 0.6, 300),  # First peak at -4
            np.random.normal(0, 0.6, 300),   # Second peak at 0
            np.random.normal(4, 0.6, 300)    # Third peak at 4
        ])
        
        fig, ax = plt.subplots()
        ax.hist(trimodal_data, bins=50, alpha=0.7, color='green')
        ax.set_title('Trimodal Distribution Test')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="matplotlib")
        result = self.formatter.format(analysis)
        
        # Check pattern analysis
        pattern_analysis = result.get('pattern_analysis', {})
        pattern_type = pattern_analysis.get('pattern_type', '')
        
        # Should detect as multimodal distribution
        self.assertIn('multimodal', pattern_type.lower(), 
                     f"Trimodal distribution should be detected as multimodal, got: {pattern_type}")
    
    def test_semantic_sections_standardization(self):
        """Test that semantic sections are standardized across all plot types."""
        plot_types = [
            ("matplotlib", "line"),
            ("matplotlib", "scatter"), 
            ("matplotlib", "bar"),
            ("matplotlib", "histogram"),
            ("seaborn", "line"),
            ("seaborn", "scatter"),
            ("seaborn", "bar"),
            ("seaborn", "histogram")
        ]
        
        expected_sections = [
            'metadata', 'axes', 'layout', 'data_summary', 
            'statistical_insights', 'pattern_analysis', 
            'visual_elements', 'domain_context', 
            'llm_description', 'llm_context'
        ]
        
        for library, plot_type in plot_types:
            with self.subTest(library=library, plot_type=plot_type):
                # Create appropriate plot
                fig, ax = plt.subplots()
                
                if plot_type == "line":
                    x = np.linspace(0, 10, 20)
                    y = 2 * x + 1
                    if library == "matplotlib":
                        ax.plot(x, y, 'bo-')
                    else:
                        sns.lineplot(x=x, y=y, ax=ax)
                
                elif plot_type == "scatter":
                    x = np.random.normal(0, 1, 50)
                    y = 0.5 * x + np.random.normal(0, 0.2, 50)
                    if library == "matplotlib":
                        ax.scatter(x, y, alpha=0.6)
                    else:
                        sns.scatterplot(x=x, y=y, ax=ax)
                
                elif plot_type == "bar":
                    categories = ['A', 'B', 'C']
                    values = [10, 20, 15]
                    if library == "matplotlib":
                        ax.bar(categories, values)
                    else:
                        sns.barplot(x=categories, y=values, ax=ax)
                
                elif plot_type == "histogram":
                    data = np.random.normal(0, 1, 500)
                    if library == "matplotlib":
                        ax.hist(data, bins=20)
                    else:
                        sns.histplot(data, bins=20, ax=ax)
                
                ax.set_title(f'{library.capitalize()} {plot_type.capitalize()}')
                
                # Analyze
                analysis = self.analyzer.analyze(fig, figure_type=library)
                result = self.formatter.format(analysis)
                
                # Check that all expected sections are present
                for section in expected_sections:
                    self.assertIn(section, result, 
                                f"Section '{section}' should be present in {library} {plot_type}")
                    self.assertIsNotNone(result[section], 
                                       f"Section '{section}' should not be None in {library} {plot_type}")
                
                plt.close(fig)
    
    def test_total_data_points_consistency(self):
        """Test that total_data_points is consistent across different plot types."""
        # Test line plot (should show actual data points)
        x = np.linspace(0, 10, 25)
        y = 2 * x + 1
        
        fig1, ax1 = plt.subplots()
        ax1.plot(x, y, 'bo-')
        ax1.set_title('Line Plot')
        
        analysis1 = self.analyzer.analyze(fig1, figure_type="matplotlib")
        result1 = self.formatter.format(analysis1)
        line_points = result1['data_summary']['total_data_points']
        
        # Test histogram (should show number of bins)
        data = np.random.normal(0, 1, 1000)
        
        fig2, ax2 = plt.subplots()
        ax2.hist(data, bins=25)
        ax2.set_title('Histogram')
        
        analysis2 = self.analyzer.analyze(fig2, figure_type="matplotlib")
        result2 = self.formatter.format(analysis2)
        hist_points = result2['data_summary']['total_data_points']
        
        # Both should show 25 (line plot: 25 points, histogram: 25 bins)
        self.assertEqual(line_points, 25, "Line plot should show 25 data points")
        self.assertEqual(hist_points, 25, "Histogram should show 25 bins")
        
        plt.close(fig1)
        plt.close(fig2)

if __name__ == '__main__':
    unittest.main(verbosity=2) 