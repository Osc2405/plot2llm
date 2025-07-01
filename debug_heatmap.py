"""
Debug script to understand why heatmap data is not being captured.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot2llm import FigureConverter

def debug_heatmap():
    """Debug heatmap detection and data extraction."""
    print("=== Debug Heatmap ===")
    
    # Create correlation matrix
    np.random.seed(42)
    data = {
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'size': np.random.uniform(10, 100, 100),
        'value': np.random.uniform(0, 1, 100)
    }
    df = pd.DataFrame(data)
    corr_matrix = df[['x', 'y', 'size', 'value']].corr()
    
    print(f"Correlation matrix shape: {corr_matrix.shape}")
    print(f"Correlation matrix:\n{corr_matrix}")
    
    # Create seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    
    # Get the figure
    fig = plt.gcf()
    
    # Debug axes and images
    print(f"\nNumber of axes: {len(fig.axes)}")
    
    for i, ax in enumerate(fig.axes):
        print(f"\nAxis {i}:")
        print(f"  Title: {ax.get_title()}")
        print(f"  Number of images: {len(ax.images)}")
        print(f"  Number of collections: {len(ax.collections)}")
        print(f"  Number of lines: {len(ax.lines)}")
        print(f"  Number of patches: {len(ax.patches)}")
        print(f"  Number of texts: {len(ax.texts)}")
        
        # Check images
        for j, image in enumerate(ax.images):
            print(f"    Image {j}:")
            print(f"      Type: {type(image)}")
            print(f"      Has get_array: {hasattr(image, 'get_array')}")
            if hasattr(image, 'get_array'):
                img_data = image.get_array()
                print(f"      Array shape: {img_data.shape if img_data is not None else 'None'}")
                print(f"      Array type: {type(img_data)}")
                if img_data is not None:
                    print(f"      Array min: {np.min(img_data)}")
                    print(f"      Array max: {np.max(img_data)}")
                    print(f"      Array mean: {np.mean(img_data)}")
    
    # Convert to LLM format
    converter = FigureConverter()
    semantic_result = converter.convert(fig, output_format="semantic")
    
    print(f"\n=== Analysis Result ===")
    print(f"Data types: {semantic_result['data_info']['data_types']}")
    print(f"Data points: {semantic_result['data_info']['data_points']}")
    
    stats = semantic_result['data_info']['statistics']
    print(f"Statistics per axis:")
    for axis in stats['per_axis']:
        print(f"  Axis {axis['axis_index']} ({axis['title']}):")
        print(f"    Data types: {axis['data_types']}")
        print(f"    Data points: {axis['data_points']}")
        print(f"    Matrix data: {axis['matrix_data'] is not None}")
        if axis['matrix_data']:
            print(f"    Matrix shape: {axis['matrix_data']['shape']}")
            print(f"    Matrix min: {axis['matrix_data']['min_value']}")
            print(f"    Matrix max: {axis['matrix_data']['max_value']}")
    
    plt.close(fig)

if __name__ == "__main__":
    debug_heatmap() 