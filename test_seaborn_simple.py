"""
Simple test script for seaborn analyzer functionality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plot2llm.analyzers.seaborn_analyzer import SeabornAnalyzer

def test_simple_seaborn():
    """Test basic seaborn analyzer functionality."""
    print("Testing Seaborn Analyzer...")
    
    # Create analyzer
    analyzer = SeabornAnalyzer()
    print("✓ Analyzer created successfully")
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'x': np.random.randn(20),
        'y': np.random.randn(20),
        'category': np.random.choice(['A', 'B'], 20)
    })
    print("✓ Sample data created")
    
    # Test 1: Simple scatter plot
    print("\n--- Test 1: Scatter Plot ---")
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=data, x='x', y='y', hue='category')
    plt.title('Test Scatter Plot')
    
    fig = plt.gcf()
    result = analyzer.analyze(fig, detail_level="low")
    
    print(f"✓ Analysis completed")
    print(f"  - Figure type: {result['basic_info']['figure_type']}")
    print(f"  - Axes count: {result['basic_info']['axes_count']}")
    print(f"  - Plot type: {result['seaborn_info']['plot_type']}")
    
    plt.close(fig)
    
    # Test 2: Heatmap
    print("\n--- Test 2: Heatmap ---")
    corr_matrix = data[['x', 'y']].corr()
    plt.figure(figsize=(4, 3))
    sns.heatmap(corr_matrix, annot=True)
    plt.title('Test Heatmap')
    
    fig = plt.gcf()
    result = analyzer.analyze(fig, detail_level="low")
    
    print(f"✓ Analysis completed")
    print(f"  - Figure type: {result['basic_info']['figure_type']}")
    print(f"  - Axes count: {result['basic_info']['axes_count']}")
    print(f"  - Plot type: {result['seaborn_info']['plot_type']}")
    
    plt.close(fig)
    
    # Test 3: FacetGrid
    print("\n--- Test 3: FacetGrid ---")
    g = sns.FacetGrid(data, col="category", height=3, aspect=1.2)
    g.map_dataframe(sns.scatterplot, x="x", y="y")
    g.fig.suptitle('Test FacetGrid')
    
    result = analyzer.analyze(g, detail_level="low")
    
    print(f"✓ Analysis completed")
    print(f"  - Figure type: {result['basic_info']['figure_type']}")
    print(f"  - Axes count: {result['basic_info']['axes_count']}")
    print(f"  - Plot type: {result['seaborn_info']['plot_type']}")
    print(f"  - Grid shape: {result['seaborn_info'].get('grid_shape', 'N/A')}")
    
    plt.close(g.fig)
    
    print("\n✓ All tests completed successfully!")

if __name__ == "__main__":
    test_simple_seaborn() 