#!/usr/bin/env python3
"""
Simple test script to directly test the seaborn_analyzer with FacetGrid and PairPlot.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from plot2llm.analyzers.seaborn_analyzer import SeabornAnalyzer

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

def test_seaborn_analyzer_direct():
    """Test the seaborn_analyzer directly."""
    
    print("=== TESTING SEABORN ANALYZER DIRECTLY ===")
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50),
        'category': np.random.choice(['A', 'B'], 50)
    })
    
    # Test 1: FacetGrid
    print("\n--- Testing FacetGrid ---")
    g_facet = sns.FacetGrid(df, col="category", height=4, aspect=1)
    g_facet.map_dataframe(sns.scatterplot, x="x", y="y")
    g_facet.fig.suptitle("FacetGrid Example")
    
    analyzer = SeabornAnalyzer()
    
    # Test _get_figure_type directly
    figure_type = analyzer._get_figure_type(g_facet)
    print(f"Direct _get_figure_type result: {figure_type}")
    
    # Test full analysis
    try:
        result = analyzer.analyze(g_facet)
        print(f"Full analysis result - figure_type: {result['basic_info']['figure_type']}")
        print(f"Full analysis result - axes_count: {result['basic_info']['axes_count']}")
        print(f"Full analysis result - title: {result['basic_info']['title']}")
    except Exception as e:
        print(f"Full analysis failed: {str(e)}")
    
    # Test 2: PairPlot
    print("\n--- Testing PairPlot ---")
    g_pair = sns.pairplot(df, hue="category", height=2)
    g_pair.fig.suptitle("PairPlot Example")
    
    # Test _get_figure_type directly
    figure_type = analyzer._get_figure_type(g_pair)
    print(f"Direct _get_figure_type result: {figure_type}")
    
    # Test full analysis
    try:
        result = analyzer.analyze(g_pair)
        print(f"Full analysis result - figure_type: {result['basic_info']['figure_type']}")
        print(f"Full analysis result - axes_count: {result['basic_info']['axes_count']}")
        print(f"Full analysis result - title: {result['basic_info']['title']}")
    except Exception as e:
        print(f"Full analysis failed: {str(e)}")
    
    # Close all figures
    plt.close('all')
    
    return g_facet, g_pair

if __name__ == "__main__":
    test_seaborn_analyzer_direct() 