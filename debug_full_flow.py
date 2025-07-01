#!/usr/bin/env python3
"""
Debug script to test the full flow from converter to analyzer for FacetGrid and PairPlot.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot2llm import FigureConverter
from plot2llm.utils import detect_figure_type

def debug_full_flow():
    """Debug the full flow for FacetGrid and PairPlot."""
    
    print("=== DEBUGGING FULL FLOW ===")
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50),
        'z': np.random.randn(50),
        'category': np.random.choice(['A', 'B'], 50)
    })
    
    # Test 1: FacetGrid
    print("\n--- Testing FacetGrid Full Flow ---")
    g_facet = sns.FacetGrid(df, col="category", height=4, aspect=1)
    g_facet.map_dataframe(sns.scatterplot, x="x", y="y")
    g_facet.fig.suptitle("FacetGrid Example")
    
    print(f"FacetGrid object type: {type(g_facet)}")
    print(f"FacetGrid class name: {g_facet.__class__.__name__}")
    print(f"FacetGrid module: {g_facet.__class__.__module__}")
    
    # Test detect_figure_type
    detected_type = detect_figure_type(g_facet)
    print(f"detect_figure_type result: {detected_type}")
    
    # Test converter
    converter = FigureConverter()
    try:
        result = converter.convert(g_facet, output_format="semantic")
        print(f"Converter result - figure_type: {result['basic_info']['figure_type']}")
        print(f"Converter result - axes_count: {result['basic_info']['axes_count']}")
        print(f"Converter result - title: {result['basic_info']['title']}")
    except Exception as e:
        print(f"Converter failed: {str(e)}")
    
    # Test 2: PairPlot
    print("\n--- Testing PairPlot Full Flow ---")
    g_pair = sns.pairplot(df, hue="category", height=2)
    g_pair.fig.suptitle("PairPlot Example")
    
    print(f"PairPlot object type: {type(g_pair)}")
    print(f"PairPlot class name: {g_pair.__class__.__name__}")
    print(f"PairPlot module: {g_pair.__class__.__module__}")
    
    # Test detect_figure_type
    detected_type = detect_figure_type(g_pair)
    print(f"detect_figure_type result: {detected_type}")
    
    # Test converter
    try:
        result = converter.convert(g_pair, output_format="semantic")
        print(f"Converter result - figure_type: {result['basic_info']['figure_type']}")
        print(f"Converter result - axes_count: {result['basic_info']['axes_count']}")
        print(f"Converter result - title: {result['basic_info']['title']}")
    except Exception as e:
        print(f"Converter failed: {str(e)}")
    
    # Test 3: Regular matplotlib figure for comparison
    print("\n--- Testing Regular Matplotlib Figure Full Flow ---")
    plt.figure(figsize=(8, 6))
    plt.scatter(df['x'], df['y'])
    plt.title("Regular Scatter Plot")
    fig = plt.gcf()
    
    print(f"Matplotlib figure type: {type(fig)}")
    print(f"Matplotlib figure class name: {fig.__class__.__name__}")
    print(f"Matplotlib figure module: {fig.__class__.__module__}")
    
    # Test detect_figure_type
    detected_type = detect_figure_type(fig)
    print(f"detect_figure_type result: {detected_type}")
    
    # Test converter
    try:
        result = converter.convert(fig, output_format="semantic")
        print(f"Converter result - figure_type: {result['basic_info']['figure_type']}")
        print(f"Converter result - axes_count: {result['basic_info']['axes_count']}")
        print(f"Converter result - title: {result['basic_info']['title']}")
    except Exception as e:
        print(f"Converter failed: {str(e)}")
    
    # Close all figures
    plt.close('all')
    
    return g_facet, g_pair, fig

if __name__ == "__main__":
    debug_full_flow() 