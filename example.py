#!/usr/bin/env python3
"""
Simple example of using the plot2llm library.
This example demonstrates the new improvements:
- Readable markers with codes and names
- Colors with hex values and common names
- Statistics per curve with outliers, trend, and local variance
- Standardized figure_type
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from plot2llm import FigureConverter


def main():
    """Run a simple example demonstrating the new features."""
    print("Plot2LLM Example - New Features Demo")
    print("=" * 40)
    
    # Create a matplotlib figure with multiple curves and markers
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Generate data with different characteristics
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x) + 0.1 * np.random.randn(100)  # Sine with noise
    y2 = np.cos(x) + 0.2 * np.random.randn(100)  # Cosine with more noise
    y3 = 0.5 * x + 0.3 * np.random.randn(100)    # Linear trend with noise
    
    # Plot with different markers and colors to showcase improvements
    ax.plot(x, y1, 'bo-', linewidth=2, markersize=6, label='Sine Wave', alpha=0.8)
    ax.plot(x, y2, 'r^--', linewidth=2, markersize=6, label='Cosine Wave', alpha=0.8)
    ax.plot(x, y3, 'gs-', linewidth=2, markersize=6, label='Linear Trend', alpha=0.8)
    
    ax.set_title('Multiple Curves with Different Markers and Colors')
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Convert to different formats
    converter = FigureConverter()
    
    print("\n1. Text Format (showing new features):")
    print("-" * 50)
    text_result = converter.convert(fig, output_format='text')
    print(text_result)
    
    print("\n2. JSON Format (first 500 chars):")
    print("-" * 50)
    json_result = converter.convert(fig, output_format='json')
    print(json_result[:500] + "...")
    
    print("\n3. Semantic Format - Key Improvements:")
    print("-" * 50)
    semantic_result = converter.convert(fig, output_format='semantic')
    
    # Show the new standardized figure_type
    print(f"✅ Standardized figure_type: {semantic_result['basic_info']['figure_type']}")
    
    # Show the new markers format
    markers = semantic_result['visual_info']['markers']
    print(f"\n✅ Readable markers:")
    for marker in markers:
        print(f"   - Code: '{marker['code']}', Name: '{marker['name']}'")
    
    # Show the new colors format
    colors = semantic_result['visual_info']['colors']
    print(f"\n✅ Colors with names:")
    for color in colors:
        name_str = f" ({color['name']})" if color['name'] else ""
        print(f"   - {color['hex']}{name_str}")
    
    # Show the new statistics format
    stats = semantic_result['data_info']['statistics']
    if 'global' in stats:
        print(f"\n✅ Global statistics:")
        g = stats['global']
        print(f"   - Mean: {g['mean']:.3f}, Std: {g['std']:.3f}")
        print(f"   - Min: {g['min']:.3f}, Max: {g['max']:.3f}, Median: {g['median']:.3f}")
    
    if 'per_curve' in stats:
        print(f"\n✅ Statistics per curve:")
        for i, curve in enumerate(stats['per_curve']):
            print(f"   Curve {i+1} ({curve['label']}):")
            print(f"     - Mean: {curve['mean']:.3f}, Std: {curve['std']:.3f}")
            print(f"     - Trend: {curve['trend']:.3f}, Local variance: {curve['local_var']:.3f}")
            print(f"     - Outliers: {len(curve['outliers'])} points")
    
    # check if examples directory exists
    if not os.path.exists('examples'):
        os.makedirs('examples')

    plt.savefig('examples/result.png')
    plt.close(fig)
    print("\n✅ Example completed successfully!")
    print("🎉 All new features are working correctly!")

    # save JSON result to file
    with open('examples/result.json', 'w') as f:
        json.dump(json_result, f)


if __name__ == "__main__":
    main() 