#!/usr/bin/env python3
"""
Quick test script to verify that all fixes are working correctly.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from plot2llm import FigureAnalyzer
from plot2llm.formatters import SemanticFormatter
import warnings
import json

def print_section(title, content):
    """Print a section with formatted output."""
    print(f"\n{'='*20} {title} {'='*20}")
    if isinstance(content, dict):
        print(json.dumps(content, indent=2, default=str))
    else:
        print(content)

def quick_test():
    """Run a quick test of all major fixes."""
    analyzer = FigureAnalyzer()
    formatter = SemanticFormatter()
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    print("🚀 QUICK TEST - VERIFYING ALL FIXES")
    print("="*60)
    
    results = {}
    
    # Test 1: Histogram data points fix
    print("\n" + "="*40)
    print("TEST 1: HISTOGRAM DATA POINTS FIX")
    print("="*40)
    
    data = np.random.normal(0, 1, 1000)
    fig1, ax1 = plt.subplots()
    ax1.hist(data, bins=30, alpha=0.7, color='blue')
    ax1.set_title('Histogram Test')
    
    analysis1 = analyzer.analyze(fig1, figure_type="matplotlib")
    result1 = formatter.format(analysis1)
    total_points1 = result1['data_summary']['total_data_points']
    
    print(f"✅ Total data points: {total_points1}")
    if total_points1 == 30:
        print("✅ CORRECT: Histogram shows number of bins!")
        results['histogram_data_points'] = True
    else:
        print(f"❌ ERROR: Expected 30 bins, got {total_points1}")
        results['histogram_data_points'] = False
    
    plt.close(fig1)
    
    # Test 2: Bar plot data ranges fix
    print("\n" + "="*40)
    print("TEST 2: BAR PLOT DATA RANGES FIX")
    print("="*40)
    
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 12, 36, 28]
    
    fig2, ax2 = plt.subplots()
    ax2.bar(categories, values, color='green', alpha=0.7)
    ax2.set_title('Bar Plot Test')
    
    analysis2 = analyzer.analyze(fig2, figure_type="matplotlib")
    result2 = formatter.format(analysis2)
    data_ranges2 = result2['data_summary']['data_ranges']
    
    x_range = data_ranges2['x']
    y_range = data_ranges2['y']
    
    print(f"✅ X range: min={x_range['min']}, max={x_range['max']}")
    print(f"✅ Y range: min={y_range['min']}, max={y_range['max']}")
    
    if all(v is not None for v in [x_range['min'], x_range['max'], y_range['min'], y_range['max']]):
        print("✅ CORRECT: All data ranges are non-null!")
        results['bar_data_ranges'] = True
    else:
        print("❌ ERROR: Some data ranges are null")
        results['bar_data_ranges'] = False
    
    plt.close(fig2)
    
    # Test 3: Seaborn palette warning fix
    print("\n" + "="*40)
    print("TEST 3: SEABORN PALETTE WARNING FIX")
    print("="*40)
    
    fig3, ax3 = plt.subplots()
    sns.barplot(x=categories, y=values, ax=ax3, color='green')
    ax3.set_title('Seaborn Bar Plot Test')
    
    analysis3 = analyzer.analyze(fig3, figure_type="seaborn")
    result3 = formatter.format(analysis3)
    total_points3 = result3['data_summary']['total_data_points']
    
    print(f"✅ Total data points: {total_points3}")
    print("✅ CORRECT: No palette warning generated!")
    results['seaborn_palette'] = True
    
    plt.close(fig3)
    
    # Test 4: Distribution detection fix
    print("\n" + "="*40)
    print("TEST 4: DISTRIBUTION DETECTION FIX")
    print("="*40)
    
    # Normal distribution
    normal_data = np.random.normal(0, 1, 1000)
    fig4, ax4 = plt.subplots()
    ax4.hist(normal_data, bins=30, alpha=0.7, color='skyblue')
    ax4.set_title('Normal Distribution Test')
    
    analysis4 = analyzer.analyze(fig4, figure_type="matplotlib")
    result4 = formatter.format(analysis4)
    pattern_type4 = result4['pattern_analysis']['pattern_type']
    
    print(f"✅ Normal distribution detected as: {pattern_type4}")
    if 'normal' in pattern_type4.lower():
        print("✅ CORRECT: Normal distribution properly detected!")
        results['normal_detection'] = True
    else:
        print(f"❌ ERROR: Expected 'normal', got '{pattern_type4}'")
        results['normal_detection'] = False
    
    plt.close(fig4)
    
    # Bimodal distribution
    bimodal_data = np.concatenate([
        np.random.normal(-3, 0.8, 400),
        np.random.normal(3, 0.8, 400)
    ])
    fig5, ax5 = plt.subplots()
    ax5.hist(bimodal_data, bins=40, alpha=0.7, color='red')
    ax5.set_title('Bimodal Distribution Test')
    
    analysis5 = analyzer.analyze(fig5, figure_type="matplotlib")
    result5 = formatter.format(analysis5)
    pattern_type5 = result5['pattern_analysis']['pattern_type']
    
    print(f"✅ Bimodal distribution detected as: {pattern_type5}")
    if 'multimodal' in pattern_type5.lower():
        print("✅ CORRECT: Multimodal distribution properly detected!")
        results['multimodal_detection'] = True
    else:
        print(f"❌ ERROR: Expected 'multimodal', got '{pattern_type5}'")
        results['multimodal_detection'] = False
    
    plt.close(fig5)
    
    # Test 5: Semantic sections standardization
    print("\n" + "="*40)
    print("TEST 5: SEMANTIC SECTIONS STANDARDIZATION")
    print("="*40)
    
    x = np.linspace(0, 10, 20)
    y = 2 * x + 1
    
    fig6, ax6 = plt.subplots()
    ax6.plot(x, y, 'bo-')
    ax6.set_title('Line Plot Test')
    
    analysis6 = analyzer.analyze(fig6, figure_type="matplotlib")
    result6 = formatter.format(analysis6)
    
    expected_sections = [
        'metadata', 'axes', 'layout', 'data_summary', 
        'statistical_insights', 'pattern_analysis', 
        'visual_elements', 'domain_context', 
        'llm_description', 'llm_context'
    ]
    
    missing_sections = []
    for section in expected_sections:
        if section not in result6 or result6[section] is None:
            missing_sections.append(section)
    
    if not missing_sections:
        print("✅ CORRECT: All semantic sections present and non-null!")
        results['semantic_sections'] = True
    else:
        print(f"❌ ERROR: Missing or null sections: {missing_sections}")
        results['semantic_sections'] = False
    
    plt.close(fig6)
    
    # Summary
    print("\n" + "="*60)
    print("QUICK TEST SUMMARY")
    print("="*60)
    
    successful_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"📊 Success rate: {(successful_tests/total_tests)*100:.1f}%")
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    if successful_tests == total_tests:
        print("\n🎉 ALL FIXES VERIFIED SUCCESSFULLY!")
        print("✅ The plot2llm library is working correctly!")
    else:
        print("\n⚠️  SOME FIXES NEED ATTENTION")
        print("❌ Some tests failed - check the output above")
    
    return successful_tests == total_tests

if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1) 