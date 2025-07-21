#!/usr/bin/env python3
"""
Quick test to verify matplotlib analyzer functionality.
"""

import matplotlib.pyplot as plt
import numpy as np
from plot2llm import convert, FigureConverter
from plot2llm.analyzers.matplotlib_analyzer import MatplotlibAnalyzer

# Turn off interactive mode
plt.ioff()


def test_basic_functionality():
    """Test basic matplotlib analyzer functionality."""
    print("Testing basic matplotlib analyzer functionality...")

    # Create a simple line plot
    fig, ax = plt.subplots()
    x = [1, 2, 3]
    y = [2, 4, 6]
    ax.plot(x, y)
    ax.set_title("Test Plot")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")

    # Test analyzer directly
    analyzer = MatplotlibAnalyzer()
    analysis = analyzer.analyze(fig)

    print(f"Figure type: {analysis['figure_type']}")
    print(f"Title: {analysis['title']}")
    print(f"Number of axes: {len(analysis['axes'])}")

    # Check axes data
    axes_data = analysis["axes"][0]
    print(f"Axis xlabel: {axes_data['xlabel']}")
    print(f"Axis ylabel: {axes_data['ylabel']}")
    print(f"Number of curve_points: {len(axes_data['curve_points'])}")

    # Check curve data
    if axes_data["curve_points"]:
        curve = axes_data["curve_points"][0]
        print(f"Curve x data: {curve['x']}")
        print(f"Curve y data: {curve['y']}")

    print("✅ Basic analyzer test passed!")
    plt.close(fig)


def test_converter():
    """Test FigureConverter functionality."""
    print("\nTesting FigureConverter...")

    # Create a simple scatter plot
    fig, ax = plt.subplots()
    x = [1, 2, 3, 4]
    y = [1, 4, 2, 3]
    ax.scatter(x, y)
    ax.set_title("Scatter Test")

    converter = FigureConverter()

    # Test text format
    text_result = converter.convert(fig, "text")
    print(f"Text format length: {len(text_result)}")
    print("Text format contains 'Scatter Test':", "Scatter Test" in text_result)

    # Test JSON format
    json_result = converter.convert(fig, "json")
    print(f"JSON format type: {type(json_result)}")
    print("JSON figure_type:", json_result.get("figure_type"))

    # Test semantic format
    semantic_result = converter.convert(fig, "semantic")
    print(f"Semantic format type: {type(semantic_result)}")
    print("Semantic figure_type:", semantic_result.get("figure_type"))

    print("✅ Converter test passed!")
    plt.close(fig)


def test_custom_formatter():
    """Test custom formatter registration."""
    print("\nTesting custom formatter...")

    class CustomFormatter:
        def format(self, analysis, **kwargs):
            return f"Custom: {analysis.get('title', 'No Title')}"

    converter = FigureConverter()
    custom_formatter = CustomFormatter()
    converter.register_formatter("custom", custom_formatter)

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.set_title("Custom Test")

    # Test using string format name
    result = converter.convert(fig, "custom")
    print(f"Custom formatter result: {result}")

    # Test using formatter object directly
    result2 = converter.convert(fig, custom_formatter)
    print(f"Direct formatter result: {result2}")

    print("✅ Custom formatter test passed!")
    plt.close(fig)


def test_different_plot_types():
    """Test different plot types."""
    print("\nTesting different plot types...")

    analyzer = MatplotlibAnalyzer()

    # Test histogram
    fig, ax = plt.subplots()
    data = np.random.normal(0, 1, 100)
    ax.hist(data)
    analysis = analyzer.analyze(fig)
    plot_types = [
        pt["type"]
        for ax_data in analysis["axes"]
        for pt in ax_data.get("plot_types", [])
    ]
    print(f"Histogram plot types: {plot_types}")
    plt.close(fig)

    # Test bar plot
    fig, ax = plt.subplots()
    categories = ["A", "B", "C"]
    values = [1, 3, 2]
    ax.bar(categories, values)
    analysis = analyzer.analyze(fig)
    plot_types = [
        pt["type"]
        for ax_data in analysis["axes"]
        for pt in ax_data.get("plot_types", [])
    ]
    print(f"Bar plot types: {plot_types}")
    plt.close(fig)

    # Test boxplot
    fig, ax = plt.subplots()
    data = np.random.normal(0, 1, 100)
    ax.boxplot(data)
    analysis = analyzer.analyze(fig)
    plot_types = [
        pt["type"]
        for ax_data in analysis["axes"]
        for pt in ax_data.get("plot_types", [])
    ]
    print(f"Boxplot plot types: {plot_types}")
    plt.close(fig)

    print("✅ Different plot types test passed!")


if __name__ == "__main__":
    print("Running quick tests for plot2llm matplotlib functionality...\n")

    try:
        test_basic_functionality()
        test_converter()
        test_custom_formatter()
        test_different_plot_types()
        print("\n🎉 All tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        plt.close("all")
