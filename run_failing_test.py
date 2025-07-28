import numpy as np
import matplotlib.pyplot as plt
from plot2llm.analyzers import FigureAnalyzer
from plot2llm.formatters import SemanticFormatter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_failing_test():
    """Run the specific failing test from test_fixes_verification.py."""
    print("🧪 RUNNING THE FAILING TEST")
    print("="*50)
    
    analyzer = FigureAnalyzer()
    formatter = SemanticFormatter()
    
    # Create the exact same bimodal distribution as in the failing test
    bimodal_data = np.concatenate([
        np.random.normal(-3, 0.8, 400),  # First peak at -3
        np.random.normal(3, 0.8, 400)    # Second peak at 3
    ])
    
    fig, ax = plt.subplots()
    ax.hist(bimodal_data, bins=40, alpha=0.7, color='red')
    ax.set_title('Bimodal Distribution Test')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    
    # Analyze
    analysis = analyzer.analyze(fig, figure_type="matplotlib")
    result = formatter.format(analysis)
    
    # Check pattern analysis
    pattern_analysis = result.get('pattern_analysis', {})
    pattern_type = pattern_analysis.get('pattern_type', '')
    
    print(f"Pattern type: {pattern_type}")
    
    # Check if it contains 'multimodal'
    success = 'multimodal' in pattern_type.lower()
    
    if success:
        print("✅ SUCCESS: Multimodal distribution correctly detected!")
        print("🎉 The fix is working!")
    else:
        print("❌ FAILURE: Multimodal distribution not detected")
        print(f"   Expected: pattern containing 'multimodal'")
        print(f"   Got: '{pattern_type}'")
        print("🔧 The fix needs further adjustment")
    
    plt.close(fig)
    
    return success

if __name__ == "__main__":
    run_failing_test() 