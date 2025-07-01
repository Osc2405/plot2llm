#!/usr/bin/env python3
"""
Debug script to analyze seaborn heatmap structure in detail.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import json

def debug_heatmap_structure():
    """Debug the structure of seaborn heatmap."""
    
    # Create sample data
    np.random.seed(42)
    data = np.random.randn(10, 10)
    corr_matrix = np.corrcoef(data)
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation Heatmap")
    
    # Get the figure and axes
    fig = plt.gcf()
    ax = plt.gca()
    
    print("=== HEATMAP DEBUG ANALYSIS ===")
    print(f"Figure type: {type(fig)}")
    print(f"Axes type: {type(ax)}")
    print(f"Number of axes: {len(fig.axes)}")
    
    # Analyze axes content
    print(f"\n=== AXES CONTENT ANALYSIS ===")
    print(f"Images: {len(ax.images)}")
    print(f"Collections: {len(ax.collections)}")
    print(f"Lines: {len(ax.lines)}")
    print(f"Patches: {len(ax.patches)}")
    print(f"Texts: {len(ax.texts)}")
    
    # Analyze images
    print(f"\n=== IMAGES ANALYSIS ===")
    for i, img in enumerate(ax.images):
        print(f"Image {i}:")
        print(f"  Type: {type(img)}")
        print(f"  Has get_array: {hasattr(img, 'get_array')}")
        if hasattr(img, 'get_array'):
            arr = img.get_array()
            print(f"  Array type: {type(arr)}")
            print(f"  Array shape: {arr.shape if hasattr(arr, 'shape') else 'No shape'}")
            print(f"  Array size: {arr.size if hasattr(arr, 'size') else 'No size'}")
            print(f"  Array min: {np.min(arr) if hasattr(arr, 'min') else 'No min'}")
            print(f"  Array max: {np.max(arr) if hasattr(arr, 'max') else 'No max'}")
    
    # Analyze collections
    print(f"\n=== COLLECTIONS ANALYSIS ===")
    for i, collection in enumerate(ax.collections):
        print(f"Collection {i}:")
        print(f"  Type: {type(collection)}")
        print(f"  Class name: {collection.__class__.__name__}")
        print(f"  Has get_array: {hasattr(collection, 'get_array')}")
        print(f"  Has get_offsets: {hasattr(collection, 'get_offsets')}")
        print(f"  Has get_facecolor: {hasattr(collection, 'get_facecolor')}")
        
        if hasattr(collection, 'get_array'):
            arr = collection.get_array()
            print(f"  Array type: {type(arr)}")
            print(f"  Array shape: {arr.shape if hasattr(arr, 'shape') else 'No shape'}")
            print(f"  Array size: {arr.size if hasattr(arr, 'size') else 'No size'}")
            if arr is not None and hasattr(arr, 'size') and arr.size > 0:
                print(f"  Array min: {np.min(arr)}")
                print(f"  Array max: {np.max(arr)}")
                print(f"  Array mean: {np.mean(arr)}")
                print(f"  Array std: {np.std(arr)}")
        
        if hasattr(collection, 'get_facecolor'):
            facecolor = collection.get_facecolor()
            print(f"  Facecolor type: {type(facecolor)}")
            print(f"  Facecolor shape: {facecolor.shape if hasattr(facecolor, 'shape') else 'No shape'}")
    
    # Analyze texts
    print(f"\n=== TEXTS ANALYSIS ===")
    for i, text in enumerate(ax.texts):
        print(f"Text {i}:")
        print(f"  Content: {text.get_text()}")
        print(f"  Position: {text.get_position()}")
    
    # Try to extract data manually
    print(f"\n=== MANUAL DATA EXTRACTION ===")
    
    # Method 1: From images
    print("Method 1: From images")
    for img in ax.images:
        if hasattr(img, 'get_array'):
            arr = img.get_array()
            if arr is not None:
                print(f"  Found data in image: shape={arr.shape}, size={arr.size}")
                print(f"  Data sample: {arr.flatten()[:10]}")
                break
    
    # Method 2: From collections (QuadMesh)
    print("Method 2: From collections (QuadMesh)")
    for collection in ax.collections:
        if collection.__class__.__name__ == "QuadMesh" and hasattr(collection, 'get_array'):
            arr = collection.get_array()
            if arr is not None:
                print(f"  Found data in QuadMesh: shape={arr.shape}, size={arr.size}")
                print(f"  Data sample: {arr.flatten()[:10]}")
                
                # Try to reshape
                n = int(np.sqrt(arr.size))
                if n * n == arr.size:
                    arr_2d = arr.reshape((n, n))
                    print(f"  Reshaped to 2D: shape={arr_2d.shape}")
                    print(f"  2D data sample: {arr_2d[:3, :3]}")
                break
    
    # Method 3: From the original data
    print("Method 3: From original data")
    print(f"  Original corr_matrix shape: {corr_matrix.shape}")
    print(f"  Original corr_matrix sample: {corr_matrix[:3, :3]}")
    
    # Save the figure
    plt.savefig('debug_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return fig, ax, corr_matrix

def debug_facetgrid_structure():
    """Debug the structure of seaborn FacetGrid."""
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Create FacetGrid
    g = sns.FacetGrid(df, col="category", height=4, aspect=1)
    g.map_dataframe(sns.scatterplot, x="x", y="y")
    g.fig.suptitle("FacetGrid Example")
    
    print("=== FACETGRID DEBUG ANALYSIS ===")
    print(f"FacetGrid type: {type(g)}")
    print(f"FacetGrid class name: {g.__class__.__name__}")
    print(f"FacetGrid module: {g.__class__.__module__}")
    
    print(f"\n=== FACETGRID ATTRIBUTES ===")
    print(f"Has fig: {hasattr(g, 'fig')}")
    print(f"Has axes: {hasattr(g, 'axes')}")
    print(f"Has figure: {hasattr(g, 'figure')}")
    
    if hasattr(g, 'fig'):
        print(f"Figure type: {type(g.fig)}")
        print(f"Number of axes in fig: {len(g.fig.axes)}")
    
    if hasattr(g, 'axes'):
        print(f"Axes type: {type(g.axes)}")
        print(f"Axes shape: {g.axes.shape if hasattr(g.axes, 'shape') else 'No shape'}")
        print(f"Number of axes: {len(g.axes.flatten()) if hasattr(g.axes, 'flatten') else len(g.axes)}")
    
    # Analyze each subplot
    if hasattr(g, 'axes'):
        axes = g.axes.flatten() if hasattr(g.axes, 'flatten') else g.axes
        for i, ax in enumerate(axes):
            print(f"\n=== SUBPLOT {i} ANALYSIS ===")
            print(f"  Type: {type(ax)}")
            print(f"  Title: {ax.get_title()}")
            print(f"  Images: {len(ax.images)}")
            print(f"  Collections: {len(ax.collections)}")
            print(f"  Lines: {len(ax.lines)}")
            print(f"  Patches: {len(ax.patches)}")
            print(f"  Texts: {len(ax.texts)}")
    
    # Save the figure
    g.fig.savefig('debug_facetgrid.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return g

def debug_pairplot_structure():
    """Debug the structure of seaborn PairPlot."""
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(50),
        'y': np.random.randn(50),
        'z': np.random.randn(50),
        'category': np.random.choice(['A', 'B'], 50)
    })
    
    # Create PairPlot
    g = sns.pairplot(df, hue="category", height=2)
    g.fig.suptitle("PairPlot Example")
    
    print("=== PAIRPLOT DEBUG ANALYSIS ===")
    print(f"PairPlot type: {type(g)}")
    print(f"PairPlot class name: {g.__class__.__name__}")
    print(f"PairPlot module: {g.__class__.__module__}")
    
    print(f"\n=== PAIRPLOT ATTRIBUTES ===")
    print(f"Has fig: {hasattr(g, 'fig')}")
    print(f"Has axes: {hasattr(g, 'axes')}")
    print(f"Has figure: {hasattr(g, 'figure')}")
    
    if hasattr(g, 'fig'):
        print(f"Figure type: {type(g.fig)}")
        print(f"Number of axes in fig: {len(g.fig.axes)}")
    
    if hasattr(g, 'axes'):
        print(f"Axes type: {type(g.axes)}")
        print(f"Axes shape: {g.axes.shape if hasattr(g.axes, 'shape') else 'No shape'}")
        print(f"Number of axes: {len(g.axes.flatten()) if hasattr(g.axes, 'flatten') else len(g.axes)}")
    
    # Analyze each subplot
    if hasattr(g, 'axes'):
        axes = g.axes.flatten() if hasattr(g.axes, 'flatten') else g.axes
        for i, ax in enumerate(axes):
            print(f"\n=== SUBPLOT {i} ANALYSIS ===")
            print(f"  Type: {type(ax)}")
            print(f"  Title: {ax.get_title()}")
            print(f"  Images: {len(ax.images)}")
            print(f"  Collections: {len(ax.collections)}")
            print(f"  Lines: {len(ax.lines)}")
            print(f"  Patches: {len(ax.patches)}")
            print(f"  Texts: {len(ax.texts)}")
    
    # Save the figure
    g.fig.savefig('debug_pairplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return g

if __name__ == "__main__":
    print("Starting detailed debug analysis...")
    
    # Debug heatmap
    print("\n" + "="*50)
    print("DEBUGGING HEATMAP")
    print("="*50)
    fig, ax, corr_matrix = debug_heatmap_structure()
    
    # Debug FacetGrid
    print("\n" + "="*50)
    print("DEBUGGING FACETGRID")
    print("="*50)
    g_facet = debug_facetgrid_structure()
    
    # Debug PairPlot
    print("\n" + "="*50)
    print("DEBUGGING PAIRPLOT")
    print("="*50)
    g_pair = debug_pairplot_structure()
    
    print("\nDebug analysis complete. Check the generated PNG files.") 