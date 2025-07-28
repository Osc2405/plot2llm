#!/usr/bin/env python3
"""
Comprehensive test runner for plot2llm library.
Runs all unit tests and provides detailed reporting.
"""

import unittest
import sys
import os
import time
import warnings
from io import StringIO

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_unit_tests():
    """Run all unit tests and return results."""
    print("="*80)
    print("RUNNING UNIT TESTS")
    print("="*80)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    if not os.path.exists(start_dir):
        print(f"❌ Test directory not found: {start_dir}")
        return None
    
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=StringIO())
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    return {
        'result': result,
        'duration': end_time - start_time,
        'output': runner.stream.getvalue()
    }

def run_integration_tests():
    """Run integration tests to verify end-to-end functionality."""
    print("\n" + "="*80)
    print("RUNNING INTEGRATION TESTS")
    print("="*80)
    
    try:
        # Import and run integration tests
        from tests.test_plot_types_unit import TestPlotTypesUnit
        
        # Create test suite for integration tests
        integration_suite = unittest.TestSuite()
        
        # Add specific integration tests
        test_cases = [
            'test_semantic_sections_completeness',
            'test_histogram_distribution_detection',
            'test_multimodal_distribution_detection'
        ]
        
        for test_case in test_cases:
            integration_suite.addTest(TestPlotTypesUnit(test_case))
        
        # Run integration tests
        runner = unittest.TextTestRunner(verbosity=2, stream=StringIO())
        start_time = time.time()
        result = runner.run(integration_suite)
        end_time = time.time()
        
        return {
            'result': result,
            'duration': end_time - start_time,
            'output': runner.stream.getvalue()
        }
        
    except ImportError as e:
        print(f"❌ Could not import integration tests: {e}")
        return None

def generate_test_report(unit_results, integration_results):
    """Generate a comprehensive test report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST REPORT")
    print("="*80)
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_duration = 0
    
    # Unit tests summary
    if unit_results:
        unit_result = unit_results['result']
        unit_tests = unit_result.testsRun
        unit_failures = len(unit_result.failures)
        unit_errors = len(unit_result.errors)
        unit_duration = unit_results['duration']
        
        total_tests += unit_tests
        total_failures += unit_failures
        total_errors += unit_errors
        total_duration += unit_duration
        
        print(f"\n📊 UNIT TESTS SUMMARY:")
        print(f"   Tests run: {unit_tests}")
        print(f"   Failures: {unit_failures}")
        print(f"   Errors: {unit_errors}")
        print(f"   Duration: {unit_duration:.2f}s")
        print(f"   Success rate: {((unit_tests - unit_failures - unit_errors) / unit_tests * 100):.1f}%")
        
        if unit_failures > 0:
            print(f"\n❌ UNIT TEST FAILURES:")
            for test, traceback in unit_result.failures:
                print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if unit_errors > 0:
            print(f"\n💥 UNIT TEST ERRORS:")
            for test, traceback in unit_result.errors:
                print(f"   - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Integration tests summary
    if integration_results:
        integration_result = integration_results['result']
        integration_tests = integration_result.testsRun
        integration_failures = len(integration_result.failures)
        integration_errors = len(integration_result.errors)
        integration_duration = integration_results['duration']
        
        total_tests += integration_tests
        total_failures += integration_failures
        total_errors += integration_errors
        total_duration += integration_duration
        
        print(f"\n📊 INTEGRATION TESTS SUMMARY:")
        print(f"   Tests run: {integration_tests}")
        print(f"   Failures: {integration_failures}")
        print(f"   Errors: {integration_errors}")
        print(f"   Duration: {integration_duration:.2f}s")
        print(f"   Success rate: {((integration_tests - integration_failures - integration_errors) / integration_tests * 100):.1f}%")
        
        if integration_failures > 0:
            print(f"\n❌ INTEGRATION TEST FAILURES:")
            for test, traceback in integration_result.failures:
                print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if integration_errors > 0:
            print(f"\n💥 INTEGRATION TEST ERRORS:")
            for test, traceback in integration_result.errors:
                print(f"   - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Overall summary
    print(f"\n🎯 OVERALL SUMMARY:")
    print(f"   Total tests: {total_tests}")
    print(f"   Total failures: {total_failures}")
    print(f"   Total errors: {total_errors}")
    print(f"   Total duration: {total_duration:.2f}s")
    
    if total_tests > 0:
        success_rate = ((total_tests - total_failures - total_errors) / total_tests) * 100
        print(f"   Overall success rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"✅ The plot2llm library is working correctly!")
        elif success_rate >= 90:
            print(f"\n⚠️  MOST TESTS PASSED")
            print(f"🔧 Some minor issues need attention")
        elif success_rate >= 75:
            print(f"\n⚠️  MANY TESTS PASSED")
            print(f"🔧 Several issues need to be addressed")
        else:
            print(f"\n❌ MANY TESTS FAILED")
            print(f"🚨 Critical issues need immediate attention")
    
    return {
        'total_tests': total_tests,
        'total_failures': total_failures,
        'total_errors': total_errors,
        'success_rate': success_rate if total_tests > 0 else 0
    }

def main():
    """Main test runner function."""
    print("🚀 PLOT2LLM COMPREHENSIVE TEST SUITE")
    print("Testing all plot types and functionality")
    print("="*80)
    
    # Run unit tests
    unit_results = run_unit_tests()
    
    # Run integration tests
    integration_results = run_integration_tests()
    
    # Generate comprehensive report
    report = generate_test_report(unit_results, integration_results)
    
    # Exit with appropriate code
    if report['total_failures'] > 0 or report['total_errors'] > 0:
        print(f"\n❌ TESTS COMPLETED WITH FAILURES")
        sys.exit(1)
    else:
        print(f"\n✅ ALL TESTS PASSED SUCCESSFULLY")
        sys.exit(0)

if __name__ == '__main__':
    main() 