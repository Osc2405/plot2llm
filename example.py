#!/usr/bin/env python3
"""
Simple example of using the plot2llm library.
"""

import matplotlib.pyplot as plt
import numpy as np
from plot2llm import FigureConverter


def main():
    """Run a simple example."""
    print("Plot2LLM Example")
    print("=" * 30)
    
    # Create a simple matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    ax.plot(x, y, 'b-', linewidth=2, label='sin(x)')
    ax.plot(x, np.cos(x), 'r--', linewidth=2, label='cos(x)')
    ax.set_title('Trigonometric Functions')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Convert to different formats
    converter = FigureConverter()
    
    print("\n1. Text Format:")
    print("-" * 20)
    text_result = converter.convert(fig, output_format='text')
    print(text_result)
    
    print("\n2. JSON Format (first 300 chars):")
    print("-" * 20)
    json_result = converter.convert(fig, output_format='json')
    print(json_result[:300] + "...")
    
    print("\n3. Semantic Format:")
    print("-" * 20)
    semantic_result = converter.convert(fig, output_format='semantic')
    print(f"Type: {type(semantic_result)}")
    print(f"Keys: {list(semantic_result.keys())}")
    print(f"Title: {semantic_result['basic_info']['title']}")
    
    plt.close(fig)
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main() 