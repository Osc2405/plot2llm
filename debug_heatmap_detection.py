#!/usr/bin/env python3
"""
Debug script to understand heatmap detection and data extraction.
"""

import sys
import os
import json

# Add the plot2llm directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'plot2llm'))

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot2llm.converter import FigureConverter
from plot2llm.utils import detect_figure_type

def debug_heatmap():
    """Debug heatmap detection and data extraction."""
    print("🔍 Debugging heatmap detection...")
    
    # Create sample data for heatmap
    data = np.random.rand(5, 5)
    
    # Create heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(data, annot=True, cmap='viridis', cbar=True)
    plt.title("Debug Heatmap")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    
    # Get the figure
    fig = plt.gcf()
    
    # Debug figure type detection
    print(f"Figure type detected: {detect_figure_type(fig)}")
    print(f"Figure class: {fig.__class__}")
    print(f"Figure module: {fig.__class__.__module__}")
    
    # Debug axes
    axes = fig.axes
    print(f"Number of axes: {len(axes)}")
    
    for i, ax in enumerate(axes):
        print(f"\nAxis {i}:")
        print(f"  Collections: {len(ax.collections)}")
        for j, collection in enumerate(ax.collections):
            print(f"    Collection {j}: {collection.__class__.__name__}")
            if hasattr(collection, 'get_array'):
                arr = collection.get_array()
                print(f"      Has array: {arr is not None}")
                if arr is not None:
                    print(f"      Array shape: {arr.shape}")
                    print(f"      Array type: {type(arr)}")
        
        print(f"  Images: {len(ax.images)}")
        for j, image in enumerate(ax.images):
            print(f"    Image {j}: {image.__class__.__name__}")
            if hasattr(image, 'get_array'):
                arr = image.get_array()
                print(f"      Has array: {arr is not None}")
                if arr is not None:
                    print(f"      Array shape: {arr.shape}")
    
    # Convert to LLM format
    converter = FigureConverter()
    result = converter.convert(fig, output_format="json")
    
    # Debug result
    print(f"\nResult figure_type: {result.get('figure_type')}")
    print(f"Result keys: {list(result.keys())}")
    
    if 'statistics' in result:
        stats = result['statistics']
        print(f"Statistics keys: {list(stats.keys())}")
        
        for i, axis_stat in enumerate(stats.get('per_axis', [])):
            print(f"\nAxis {i} statistics:")
            print(f"  Matrix data: {axis_stat.get('matrix_data') is not None}")
            print(f"  Data points: {axis_stat.get('data_points')}")
            print(f"  Data types: {axis_stat.get('data_types')}")
            if axis_stat.get('matrix_data'):
                matrix = axis_stat['matrix_data']
                print(f"  Matrix shape: {matrix.get('shape')}")
                print(f"  Matrix min: {matrix.get('min_value')}")
                print(f"  Matrix max: {matrix.get('max_value')}")
    
    # Save result
    with open('debug_heatmap_result.json', 'w', encoding='utf-8') as f:
        json_string = converter.json_formatter.to_string(result)
        f.write(json_string)
    
    print("\n✅ Debug result saved to debug_heatmap_result.json")
    
    plt.close()

if __name__ == "__main__":
    debug_heatmap() 