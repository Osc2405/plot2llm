"""
Simple test to verify basic functionality.
"""

import matplotlib.pyplot as plt
from python2llm import FigureConverter


def test_basic_conversion():
    """Test basic conversion of a matplotlib figure."""
    # Create a simple matplotlib figure
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6], 'ro-', label='Test Data')
    ax.set_title('Test Plot')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.grid(True)
    ax.legend()
    
    # Convert to text
    converter = FigureConverter()
    result = converter.convert(fig, output_format='text')
    
    # Basic assertions
    assert isinstance(result, str)
    assert 'Figure type: matplotlib.figure' in result
    assert 'Title: Test Plot' in result
    assert 'X Axis' in result
    assert 'Y Axis' in result
    
    print("✅ Basic conversion test passed!")
    print(f"Result preview: {result[:200]}...")
    
    plt.close(fig)


def test_json_conversion():
    """Test JSON conversion."""
    # Create a simple figure
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6])
    ax.set_title('Scatter Plot')
    
    # Convert to JSON
    converter = FigureConverter()
    result = converter.convert(fig, output_format='json')
    
    # Basic assertions
    assert isinstance(result, str)
    import json
    parsed = json.loads(result)
    assert 'basic_info' in parsed
    assert 'axes_info' in parsed
    
    print("✅ JSON conversion test passed!")
    
    plt.close(fig)


if __name__ == "__main__":
    print("Running basic tests...")
    test_basic_conversion()
    test_json_conversion()
    print("All tests passed! 🎉") 