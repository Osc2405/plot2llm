import numpy as np
import matplotlib.pyplot as plt
from plot2llm.analyzers import FigureAnalyzer
from plot2llm.formatters import SemanticFormatter
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def run_original_failing_test():
    """Run the exact test that was failing in test_fixes_verification.py."""
    print("🧪 RUNNING ORIGINAL FAILING TEST")
    print("="*50)
    
    analyzer = FigureAnalyzer()
    formatter = SemanticFormatter()
    
    # Test 1: Normal distribution (this was the failing test)
    print("\n📊 Test: Normal Distribution Detection")
    print("-" * 40)
    
    # Create a normal distribution (exact same as in the failing test)
    normal_data = np.random.normal(0, 1, 1000)
    
    fig, ax = plt.subplots()
    ax.hist(normal_data, bins=30, alpha=0.7, color='skyblue')
    ax.set_title('Normal Distribution Test')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    
    # Analyze
    analysis = analyzer.analyze(fig, figure_type="matplotlib")
    result = formatter.format(analysis)
    
    # Check pattern analysis
    pattern_analysis = result.get('pattern_analysis', {})
    pattern_type = pattern_analysis.get('pattern_type', '')
    
    print(f"Pattern type: {pattern_type}")
    
    # Check if it contains 'normal' and NOT 'multimodal'
    success = 'normal' in pattern_type.lower() and 'multimodal' not in pattern_type.lower()
    
    if success:
        print("✅ SUCCESS: Normal distribution correctly detected!")
        print("🎉 The fix is working!")
    else:
        print("❌ FAILURE: Normal distribution not detected correctly")
        print(f"   Expected: pattern containing 'normal' and NOT 'multimodal'")
        print(f"   Got: '{pattern_type}'")
        print("🔧 The fix needs further adjustment")
    
    plt.close(fig)
    
    # Test 2: Bimodal distribution (this should still work)
    print("\n📊 Test: Bimodal Distribution Detection")
    print("-" * 40)
    
    bimodal_data = np.concatenate([
        np.random.normal(-3, 0.8, 400),  # First peak at -3
        np.random.normal(3, 0.8, 400)    # Second peak at 3
    ])
    
    fig2, ax2 = plt.subplots()
    ax2.hist(bimodal_data, bins=40, alpha=0.7, color='red')
    ax2.set_title('Bimodal Distribution Test')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    
    analysis2 = analyzer.analyze(fig2, figure_type="matplotlib")
    result2 = formatter.format(analysis2)
    
    pattern_type2 = result2.get('pattern_analysis', {}).get('pattern_type', '')
    print(f"Pattern type: {pattern_type2}")
    
    success2 = 'multimodal' in pattern_type2.lower()
    
    if success2:
        print("✅ SUCCESS: Multimodal distribution correctly detected!")
    else:
        print("❌ FAILURE: Multimodal distribution not detected")
        print(f"   Expected: pattern containing 'multimodal'")
        print(f"   Got: '{pattern_type2}'")
    
    plt.close(fig2)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    if success and success2:
        print("🎉 BOTH TESTS PASSED!")
        print("✅ The distribution detection is now balanced!")
        print("✅ Normal distributions are detected as normal")
        print("✅ Multimodal distributions are detected as multimodal")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("❌ Further adjustment needed")
    
    return success and success2

if __name__ == "__main__":
    run_original_failing_test() 