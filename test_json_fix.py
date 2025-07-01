#!/usr/bin/env python3
"""
Simple test to verify JSONFormatter fix.
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

def test_json_formatter():
    """Test that JSONFormatter returns a dictionary, not a string."""
    print("🔍 Testing JSONFormatter fix...")
    
    # Create a simple heatmap
    data = np.random.rand(5, 5)
    plt.figure(figsize=(6, 4))
    sns.heatmap(data, annot=True, cmap='viridis')
    plt.title("Simple Heatmap")
    
    # Convert to JSON format
    converter = FigureConverter()
    result = converter.convert(plt.gcf(), output_format="json")
    
    # Check the type of result
    print(f"Result type: {type(result)}")
    print(f"Is dict: {isinstance(result, dict)}")
    
    if isinstance(result, dict):
        print("✅ JSONFormatter fix: SUCCESS - Returns dictionary")
        print(f"Result keys: {list(result.keys())}")
        
        # Check if statistics are present
        if 'statistics' in result:
            print("✅ Statistics key found")
            stats = result['statistics']
            if 'per_axis' in stats:
                print(f"✅ Per-axis statistics: {len(stats['per_axis'])} axes")
                for i, axis_stat in enumerate(stats['per_axis']):
                    if 'matrix_data' in axis_stat:
                        print(f"✅ Axis {i}: Matrix data found")
                    if 'mean' in axis_stat and axis_stat['mean'] is not None:
                        print(f"✅ Axis {i}: Statistics found")
        
        # Check if axis_info is present
        if 'axis_info' in result:
            print("✅ Axis info found")
            axis_info = result['axis_info']
            print(f"Total axes: {axis_info.get('total_axes', 0)}")
            print(f"Figure title: {axis_info.get('figure_title', 'Not found')}")
        
        # Save result
        with open('test_json_fix_result.json', 'w', encoding='utf-8') as f:
            # Use the formatter's to_string method to handle numpy arrays
            json_string = converter.json_formatter.to_string(result)
            f.write(json_string)
        print("✅ Result saved to test_json_fix_result.json")
        
    else:
        print("❌ JSONFormatter fix: FAILED - Still returns string")
        print(f"Result: {result[:200]}...")  # Show first 200 chars
    
    plt.close()

if __name__ == "__main__":
    test_json_formatter() 