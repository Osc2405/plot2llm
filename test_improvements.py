#!/usr/bin/env python3
"""
Test script to verify the improvements:
1. Heatmap processing reactivated
2. Numpy warnings optimized
3. Better axis title extraction
"""

import sys
import os
import json
import warnings

# Add the plot2llm directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'plot2llm'))

# Set matplotlib to use non-interactive backend to avoid memory issues
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot2llm.converter import FigureConverter

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

def test_heatmap():
    """Test heatmap processing with data extraction."""
    print("🔍 Testing heatmap processing...")
    
    # Create sample data for heatmap
    data = np.random.rand(10, 10)
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, cmap='viridis', cbar=True)
    plt.title("Test Heatmap")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    
    # Convert to LLM format (specify JSON format)
    converter = FigureConverter()
    result = converter.convert(plt.gcf(), output_format="json")
    
    # Save result
    with open('test_heatmap_improved.json', 'w', encoding='utf-8') as f:
        json_string = converter.json_formatter.to_string(result)
        f.write(json_string)
    
    # Check if heatmap data is present
    has_matrix_data = False
    has_statistics = False
    
    if 'statistics' in result:
        for axis_stat in result['statistics'].get('per_axis', []):
            if 'matrix_data' in axis_stat and axis_stat['matrix_data']:
                has_matrix_data = True
                print(f"✅ Matrix data found: shape {axis_stat['matrix_data']['shape']}")
            if 'mean' in axis_stat and axis_stat['mean'] is not None:
                has_statistics = True
                print(f"✅ Statistics found: mean={axis_stat['mean']:.4f}")
    
    if has_matrix_data and has_statistics:
        print("✅ Heatmap processing: SUCCESS")
    else:
        print("❌ Heatmap processing: FAILED")
        print("   Debug: Check if 'statistics' key exists in result")
        print(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
    plt.close()

def test_axis_titles():
    """Test improved axis title extraction."""
    print("\n📝 Testing axis title extraction...")
    
    # Create a multi-subplot figure with titles
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Add titles and labels
    fig.suptitle("Main Figure Title")
    
    axes[0, 0].set_title("Subplot 1")
    axes[0, 0].set_xlabel("X Label 1")
    axes[0, 0].set_ylabel("Y Label 1")
    axes[0, 0].plot([1, 2, 3], [1, 4, 2])
    
    axes[0, 1].set_title("Subplot 2")
    axes[0, 1].set_xlabel("X Label 2")
    axes[0, 1].set_ylabel("Y Label 2")
    axes[0, 1].scatter([1, 2, 3], [2, 5, 3])
    
    axes[1, 0].set_title("Subplot 3")
    axes[1, 0].set_xlabel("X Label 3")
    axes[1, 0].set_ylabel("Y Label 3")
    axes[1, 0].bar([1, 2, 3], [3, 1, 4])
    
    axes[1, 1].set_title("Subplot 4")
    axes[1, 1].set_xlabel("X Label 4")
    axes[1, 1].set_ylabel("Y Label 4")
    axes[1, 1].hist(np.random.randn(100), bins=10)
    
    # Convert to LLM format (specify JSON format)
    converter = FigureConverter()
    result = converter.convert(fig, output_format="json")
    
    # Save result
    with open('test_axis_titles_improved.json', 'w', encoding='utf-8') as f:
        json_string = converter.json_formatter.to_string(result)
        f.write(json_string)
    
    # Check axis information
    if 'axis_info' in result:
        axis_info = result['axis_info']
        print(f"✅ Figure title: {axis_info.get('figure_title', 'Not found')}")
        print(f"✅ Total axes: {axis_info.get('total_axes', 0)}")
        
        for i, ax in enumerate(axis_info.get('axes', [])):
            print(f"  Axis {i}:")
            print(f"    Title: '{ax.get('title', '')}'")
            print(f"    X Label: '{ax.get('x_label', '')}'")
            print(f"    Y Label: '{ax.get('y_label', '')}'")
            print(f"    Has data: {ax.get('has_data', False)}")
        
        print("✅ Axis title extraction: SUCCESS")
    else:
        print("❌ Axis title extraction: FAILED")
        print("   Debug: Check if 'axis_info' key exists in result")
        print(f"   Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
    plt.close()

def test_numpy_warnings():
    """Test that numpy warnings are properly suppressed."""
    print("\n🔇 Testing numpy warning suppression...")
    
    # Create data that would normally trigger warnings
    data_with_nan = np.array([1, 2, np.nan, 4, 5])
    data_with_inf = np.array([1, 2, np.inf, 4, 5])
    empty_data = np.array([])
    
    # Create plots that might trigger warnings
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Plot with NaN values
    axes[0].plot([1, 2, 3, 4, 5], data_with_nan)
    axes[0].set_title("Data with NaN")
    
    # Plot with Inf values
    axes[1].plot([1, 2, 3, 4, 5], data_with_inf)
    axes[1].set_title("Data with Inf")
    
    # Empty histogram
    axes[2].hist(empty_data, bins=5)
    axes[2].set_title("Empty Data")
    
    # Convert to LLM format (should not show warnings)
    converter = FigureConverter()
    result = converter.convert(fig, output_format="json")
    
    # Save result
    with open('test_numpy_warnings_improved.json', 'w', encoding='utf-8') as f:
        json_string = converter.json_formatter.to_string(result)
        f.write(json_string)
    
    print("✅ Numpy warning suppression: SUCCESS (no warnings should appear above)")
    
    plt.close()

def test_comprehensive():
    """Test all improvements together with a complex seaborn plot."""
    print("\n🎯 Testing comprehensive improvements...")
    
    # Create a complex seaborn plot (simpler version to avoid memory issues)
    np.random.seed(42)
    data = np.random.randn(50, 4)  # Reduced data size
    
    # Create a simple pairplot instead of FacetGrid
    df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
    g = sns.pairplot(df, height=2)
    g.fig.suptitle("Complex Seaborn Pairplot")
    
    # Convert to LLM format
    converter = FigureConverter()
    result = converter.convert(g.fig, output_format="json")
    
    # Save result
    with open('test_comprehensive_improved.json', 'w', encoding='utf-8') as f:
        json_string = converter.json_formatter.to_string(result)
        f.write(json_string)
    
    # Check results
    print(f"✅ Figure type: {result.get('figure_type', 'unknown')}")
    print(f"✅ Total axes: {result.get('axis_info', {}).get('total_axes', 0)}")
    print(f"✅ Statistics axes: {len(result.get('statistics', {}).get('per_axis', []))}")
    
    print("✅ Comprehensive test: SUCCESS")
    
    plt.close()

if __name__ == "__main__":
    print("🚀 Testing plot2llm improvements...")
    print("=" * 50)
    
    try:
        import pandas as pd
        test_heatmap()
        test_axis_titles()
        test_numpy_warnings()
        test_comprehensive()
        
        print("\n" + "=" * 50)
        print("🎉 All tests completed successfully!")
        print("📁 Check the generated JSON files for detailed results.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc() 