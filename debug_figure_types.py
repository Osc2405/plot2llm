#!/usr/bin/env python3
"""
Debug script to understand why FacetGrid and PairPlot are not detected correctly.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot2llm.analyzers.seaborn_analyzer import SeabornAnalyzer

def debug_figure_type_detection():
    """Debug the figure type detection for seaborn objects."""
    
    print("=== DEBUGGING FIGURE TYPE DETECTION ===")
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50),
        'z': np.random.randn(50),
        'category': np.random.choice(['A', 'B'], 50)
    })
    
    # Test 1: FacetGrid
    print("\n--- Testing FacetGrid ---")
    g_facet = sns.FacetGrid(df, col="category", height=4, aspect=1)
    g_facet.map_dataframe(sns.scatterplot, x="x", y="y")
    g_facet.fig.suptitle("FacetGrid Example")
    
    print(f"FacetGrid object type: {type(g_facet)}")
    print(f"FacetGrid class name: {g_facet.__class__.__name__}")
    print(f"FacetGrid module: {g_facet.__class__.__module__}")
    
    # Test the analyzer
    analyzer = SeabornAnalyzer()
    figure_type = analyzer._get_figure_type(g_facet)
    print(f"Analyzer detected type: {figure_type}")
    
    # Test 2: PairPlot
    print("\n--- Testing PairPlot ---")
    g_pair = sns.pairplot(df, hue="category", height=2)
    g_pair.fig.suptitle("PairPlot Example")
    
    print(f"PairPlot object type: {type(g_pair)}")
    print(f"PairPlot class name: {g_pair.__class__.__name__}")
    print(f"PairPlot module: {g_pair.__class__.__module__}")
    
    # Test the analyzer
    figure_type = analyzer._get_figure_type(g_pair)
    print(f"Analyzer detected type: {figure_type}")
    
    # Test 3: Regular matplotlib figure
    print("\n--- Testing Regular Matplotlib Figure ---")
    plt.figure(figsize=(8, 6))
    plt.scatter(df['x'], df['y'])
    plt.title("Regular Scatter Plot")
    fig = plt.gcf()
    
    print(f"Matplotlib figure type: {type(fig)}")
    print(f"Matplotlib figure class name: {fig.__class__.__name__}")
    print(f"Matplotlib figure module: {fig.__class__.__module__}")
    
    # Test the analyzer
    figure_type = analyzer._get_figure_type(fig)
    print(f"Analyzer detected type: {figure_type}")
    
    # Test 4: Full analysis of FacetGrid
    print("\n--- Full Analysis of FacetGrid ---")
    try:
        result = analyzer.analyze(g_facet)
        print(f"Analysis successful: {result['basic_info']['figure_type']}")
        print(f"Number of axes: {result['basic_info']['axes_count']}")
        print(f"Title: {result['basic_info']['title']}")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
    
    # Test 5: Full analysis of PairPlot
    print("\n--- Full Analysis of PairPlot ---")
    try:
        result = analyzer.analyze(g_pair)
        print(f"Analysis successful: {result['basic_info']['figure_type']}")
        print(f"Number of axes: {result['basic_info']['axes_count']}")
        print(f"Title: {result['basic_info']['title']}")
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
    
    # Close all figures
    plt.close('all')
    
    return g_facet, g_pair, fig

if __name__ == "__main__":
    debug_figure_type_detection() 