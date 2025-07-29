import unittest
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plot2llm import FigureAnalyzer
from plot2llm.formatters import SemanticFormatter
import warnings
import json

class TestPlotTypesUnit(unittest.TestCase):
    """Unit tests for all plot types in both matplotlib and seaborn."""
    
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
    
    def assert_data_summary_complete(self, data_summary, plot_type):
        """Assert that data_summary has all required fields."""
        self.assertIsNotNone(data_summary, f"data_summary should not be None for {plot_type}")
        
        # Check total_data_points
        self.assertIsNotNone(data_summary.get('total_data_points'), 
                            f"total_data_points should not be None for {plot_type}")
        self.assertGreater(data_summary.get('total_data_points', 0), 0,
                          f"total_data_points should be greater than 0 for {plot_type}")
        
        # Check data_ranges
        data_ranges = data_summary.get('data_ranges')
        self.assertIsNotNone(data_ranges, f"data_ranges should not be None for {plot_type}")
        
        # Check x range
        x_range = data_ranges.get('x')
        self.assertIsNotNone(x_range, f"data_ranges.x should not be None for {plot_type}")
        self.assertIsNotNone(x_range.get('min'), f"data_ranges.x.min should not be None for {plot_type}")
        self.assertIsNotNone(x_range.get('max'), f"data_ranges.x.max should not be None for {plot_type}")
        self.assertIsNotNone(x_range.get('type'), f"data_ranges.x.type should not be None for {plot_type}")
        
        # Check y range
        y_range = data_ranges.get('y')
        self.assertIsNotNone(y_range, f"data_ranges.y should not be None for {plot_type}")
        self.assertIsNotNone(y_range.get('min'), f"data_ranges.y.min should not be None for {plot_type}")
        self.assertIsNotNone(y_range.get('max'), f"data_ranges.y.max should not be None for {plot_type}")
        self.assertIsNotNone(y_range.get('type'), f"data_ranges.y.type should not be None for {plot_type}")
        
        # Check missing_values
        missing_values = data_summary.get('missing_values')
        self.assertIsNotNone(missing_values, f"missing_values should not be None for {plot_type}")
        self.assertIsNotNone(missing_values.get('x'), f"missing_values.x should not be None for {plot_type}")
        self.assertIsNotNone(missing_values.get('y'), f"missing_values.y should not be None for {plot_type}")
        
        # Check types
        self.assertIsNotNone(data_summary.get('x_type'), f"x_type should not be None for {plot_type}")
        self.assertIsNotNone(data_summary.get('y_type'), f"y_type should not be None for {plot_type}")
    
    def test_matplotlib_line_plot(self):
        """Test matplotlib line plot analysis."""
        # Create line plot
        x = np.linspace(0, 10, 50)
        y = 2 * x + 1 + np.random.normal(0, 0.5, 50)
        
        fig, ax = plt.subplots()
        ax.plot(x, y, 'bo-', label='Test Line')
        ax.set_title('Matplotlib Line Plot')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.grid(True)
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="matplotlib")
        result = self.formatter.format(analysis)
        
        # Assertions
        self.assertIn('data_summary', result)
        data_summary = result['data_summary']
        
        # Check data_summary completeness
        self.assert_data_summary_complete(data_summary, "matplotlib_line")
        
        # Specific assertions for line plot
        self.assertEqual(data_summary['total_data_points'], 50)
        self.assertEqual(data_summary['x_type'], 'numeric')
        self.assertEqual(data_summary['y_type'], 'numeric')
        self.assertEqual(data_summary['data_ranges']['x']['type'], 'numeric')
        self.assertEqual(data_summary['data_ranges']['y']['type'], 'numeric')
        
        # Check that ranges are reasonable
        x_range = data_summary['data_ranges']['x']
        self.assertAlmostEqual(x_range['min'], 0.0, places=0)  # Allow for slight extension
        self.assertAlmostEqual(x_range['max'], 10.0, places=0)  # Allow for slight extension
    
    def test_matplotlib_scatter_plot(self):
        """Test matplotlib scatter plot analysis."""
        # Create scatter plot
        x_scatter = np.random.normal(0, 1, 100)
        y_scatter = 0.5 * x_scatter + np.random.normal(0, 0.3, 100)
        
        fig, ax = plt.subplots()
        ax.scatter(x_scatter, y_scatter, alpha=0.6, color='red')
        ax.set_title('Matplotlib Scatter Plot')
        ax.set_xlabel('X Values')
        ax.set_ylabel('Y Values')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="matplotlib")
        result = self.formatter.format(analysis)
        
        # Assertions
        self.assertIn('data_summary', result)
        data_summary = result['data_summary']
        
        # Check data_summary completeness
        self.assert_data_summary_complete(data_summary, "matplotlib_scatter")
        
        # Specific assertions for scatter plot
        self.assertEqual(data_summary['total_data_points'], 100)
        self.assertEqual(data_summary['x_type'], 'numeric')
        self.assertEqual(data_summary['y_type'], 'numeric')
    
    def test_matplotlib_bar_plot(self):
        """Test matplotlib bar plot analysis."""
        # Create bar plot
        categories = ['A', 'B', 'C', 'D', 'E']
        values = [23, 45, 12, 36, 28]
        
        fig, ax = plt.subplots()
        ax.bar(categories, values, color='green', alpha=0.7)
        ax.set_title('Matplotlib Bar Plot')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Values')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="matplotlib")
        result = self.formatter.format(analysis)
        
        # Assertions
        self.assertIn('data_summary', result)
        data_summary = result['data_summary']
        
        # Check data_summary completeness
        self.assert_data_summary_complete(data_summary, "matplotlib_bar")
        
        # Specific assertions for bar plot
        self.assertEqual(data_summary['total_data_points'], 5)
        self.assertEqual(data_summary['x_type'], 'categorical')
        self.assertEqual(data_summary['y_type'], 'numeric')
        self.assertEqual(data_summary['data_ranges']['x']['type'], 'categorical')
        self.assertEqual(data_summary['data_ranges']['y']['type'], 'numeric')
    
    def test_matplotlib_histogram(self):
        """Test matplotlib histogram analysis."""
        # Create histogram
        data_hist = np.random.normal(0, 1, 1000)
        
        fig, ax = plt.subplots()
        ax.hist(data_hist, bins=30, alpha=0.7, color='purple')
        ax.set_title('Matplotlib Histogram')
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="matplotlib")
        result = self.formatter.format(analysis)
        
        # Assertions
        self.assertIn('data_summary', result)
        data_summary = result['data_summary']
        
        # Check data_summary completeness
        self.assert_data_summary_complete(data_summary, "matplotlib_histogram")
        
        # Specific assertions for histogram
        self.assertEqual(data_summary['total_data_points'], 30)  # Number of bins
        self.assertEqual(data_summary['x_type'], 'numeric')
        self.assertEqual(data_summary['y_type'], 'numeric')
        self.assertEqual(data_summary['data_ranges']['x']['type'], 'numeric')
        self.assertEqual(data_summary['data_ranges']['y']['type'], 'numeric')
    
    def test_seaborn_line_plot(self):
        """Test seaborn line plot analysis."""
        # Create line plot
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + np.random.normal(0, 0.1, 50)
        
        fig, ax = plt.subplots()
        sns.lineplot(x=x, y=y, ax=ax, color='blue')
        ax.set_title('Seaborn Line Plot')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="seaborn")
        result = self.formatter.format(analysis)
        
        # Assertions
        self.assertIn('data_summary', result)
        data_summary = result['data_summary']
        
        # Check data_summary completeness
        self.assert_data_summary_complete(data_summary, "seaborn_line")
        
        # Specific assertions for line plot
        self.assertEqual(data_summary['total_data_points'], 50)
        self.assertEqual(data_summary['x_type'], 'numeric')
        self.assertEqual(data_summary['y_type'], 'numeric')
    
    def test_seaborn_scatter_plot(self):
        """Test seaborn scatter plot analysis."""
        # Create scatter plot
        x_scatter = np.random.normal(0, 1, 100)
        y_scatter = 0.3 * x_scatter + np.random.normal(0, 0.2, 100)
        
        fig, ax = plt.subplots()
        sns.scatterplot(x=x_scatter, y=y_scatter, ax=ax, color='red', alpha=0.6)
        ax.set_title('Seaborn Scatter Plot')
        ax.set_xlabel('X Values')
        ax.set_ylabel('Y Values')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="seaborn")
        result = self.formatter.format(analysis)
        
        # Assertions
        self.assertIn('data_summary', result)
        data_summary = result['data_summary']
        
        # Check data_summary completeness
        self.assert_data_summary_complete(data_summary, "seaborn_scatter")
        
        # Specific assertions for scatter plot
        self.assertEqual(data_summary['total_data_points'], 100)
        self.assertEqual(data_summary['x_type'], 'numeric')
        self.assertEqual(data_summary['y_type'], 'numeric')
    
    def test_seaborn_bar_plot(self):
        """Test seaborn bar plot analysis."""
        # Create bar plot
        categories = ['A', 'B', 'C', 'D', 'E']
        values = [15, 32, 18, 45, 22]
        
        fig, ax = plt.subplots()
        sns.barplot(x=categories, y=values, ax=ax, color='green')
        ax.set_title('Seaborn Bar Plot')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Values')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="seaborn")
        result = self.formatter.format(analysis)
        
        # Assertions
        self.assertIn('data_summary', result)
        data_summary = result['data_summary']
        
        # Check data_summary completeness
        self.assert_data_summary_complete(data_summary, "seaborn_bar")
        
        # Specific assertions for bar plot
        self.assertEqual(data_summary['total_data_points'], 5)
        self.assertEqual(data_summary['x_type'], 'categorical')
        self.assertEqual(data_summary['y_type'], 'numeric')
        self.assertEqual(data_summary['data_ranges']['x']['type'], 'categorical')
        self.assertEqual(data_summary['data_ranges']['y']['type'], 'numeric')
    
    def test_seaborn_histogram(self):
        """Test seaborn histogram analysis."""
        # Create histogram with fixed seed for reproducibility
        np.random.seed(42)
        data_hist = np.random.normal(0, 1, 1000)
        
        fig, ax = plt.subplots()
        sns.histplot(data_hist, bins=30, kde=False, ax=ax, color='purple')
        ax.set_title('Seaborn Histogram')
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="seaborn")
        result = self.formatter.format(analysis)
        
        # Assertions
        self.assertIn('data_summary', result)
        data_summary = result['data_summary']
        
        # Check data_summary completeness
        self.assert_data_summary_complete(data_summary, "seaborn_histogram")
        
        # Specific assertions for histogram
        self.assertEqual(data_summary['total_data_points'], 30)  # Number of bins
        self.assertEqual(data_summary['x_type'], 'numeric')
        self.assertEqual(data_summary['y_type'], 'numeric')
        self.assertEqual(data_summary['data_ranges']['x']['type'], 'numeric')
        self.assertEqual(data_summary['data_ranges']['y']['type'], 'numeric')
    
    def test_semantic_sections_completeness(self):
        """Test that all semantic sections are present and complete."""
        # Create a simple line plot for testing
        x = np.linspace(0, 10, 20)
        y = 2 * x + 1
        
        fig, ax = plt.subplots()
        ax.plot(x, y, 'bo-')
        ax.set_title('Test Plot')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Analyze
        analysis = self.analyzer.analyze(fig, figure_type="matplotlib")
        result = self.formatter.format(analysis)
        
        # Check that all expected sections are present
        expected_sections = [
            'metadata', 'axes', 'layout', 'data_summary', 
            'statistical_insights', 'pattern_analysis', 
            'visual_elements', 'domain_context', 
            'llm_description', 'llm_context'
        ]
        
        for section in expected_sections:
            self.assertIn(section, result, f"Section '{section}' should be present in semantic output")
            self.assertIsNotNone(result[section], f"Section '{section}' should not be None")
    
    def test_histogram_distribution_detection(self):
        """Test histogram distribution detection accuracy."""
        # Create normal distribution with fixed seed for reproducibility
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 1000)
        
        fig, ax = plt.subplots()
        ax.hist(normal_data, bins=30, alpha=0.7, color='purple')
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
    
    def test_multimodal_distribution_detection(self):
        """Test multimodal distribution detection."""
        # Create bimodal distribution with fixed seed for reproducibility
        np.random.seed(123)
        bimodal_data = np.concatenate([
            np.random.normal(-3, 0.8, 400),
            np.random.normal(3, 0.8, 400)
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
    
    def test_trimodal_distribution_detection(self):
        """Test trimodal distribution detection."""
        # Create trimodal distribution with fixed seed for reproducibility
        np.random.seed(456)
        trimodal_data = np.concatenate([
            np.random.normal(-4, 0.6, 300),
            np.random.normal(0, 0.6, 300),
            np.random.normal(4, 0.6, 300)
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

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 