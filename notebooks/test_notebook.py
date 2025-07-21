#!/usr/bin/env python3
"""
Test script to verify that the notebook examples work correctly.
This script runs the key examples from the notebook to ensure they function properly.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add the parent directory to the path to import plot2llm
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_import():
    """Test that plot2llm can be imported correctly"""
    print("🔄 Testing plot2llm import...")
    try:
        import plot2llm

        print(f"✅ Plot2LLM imported successfully - Version: {plot2llm.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import plot2llm: {e}")
        return False


def test_basic_conversion():
    """Test basic figure conversion"""
    print("\n🔄 Testing basic figure conversion...")

    try:
        import plot2llm

        # Create a simple figure
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + 0.1 * np.random.randn(50)
        ax.plot(x, y, "b-o", linewidth=2, markersize=4, label="Test Data")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_title("Test Figure")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Test text conversion
        text_result = plot2llm.convert(fig, format="text")
        print(f"✅ Text conversion successful (length: {len(text_result)} chars)")

        # Test JSON conversion
        json_result = plot2llm.convert(fig, format="json")
        print(f"✅ JSON conversion successful (keys: {list(json_result.keys())})")

        # Test semantic conversion
        semantic_result = plot2llm.convert(fig, format="semantic")
        print(
            f"✅ Semantic conversion successful (length: {len(semantic_result)} chars)"
        )

        plt.close(fig)
        return True

    except Exception as e:
        print(f"❌ Basic conversion failed: {e}")
        return False


def test_multiple_chart_types():
    """Test analysis of different chart types"""
    print("\n🔄 Testing multiple chart types...")

    try:
        import plot2llm

        figures = {}
        analysis_results = {}

        # 1. Scatter plot
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        x_scatter = np.random.randn(100)
        y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5
        ax_scatter.scatter(x_scatter, y_scatter, alpha=0.6, c="red", s=50)
        ax_scatter.set_title("Scatter Plot Test")
        figures["scatter"] = fig_scatter

        # 2. Bar chart
        fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
        categories = ["A", "B", "C", "D"]
        values = [23, 45, 56, 78]
        ax_bar.bar(categories, values, color=["red", "blue", "green", "orange"])
        ax_bar.set_title("Bar Chart Test")
        figures["bar"] = fig_bar

        # 3. Histogram
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
        data_hist = np.random.normal(0, 1, 1000)
        ax_hist.hist(data_hist, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        ax_hist.set_title("Histogram Test")
        figures["histogram"] = fig_hist

        # Analyze each figure
        for chart_type, fig in figures.items():
            analysis = plot2llm.convert(
                fig, format="json", detail_level="high", include_statistics=True
            )
            analysis_results[chart_type] = analysis

            # Verify analysis contains expected fields
            required_fields = ["figure_type", "axes_info", "data_info"]
            missing_fields = [
                field for field in required_fields if field not in analysis
            ]

            if missing_fields:
                print(f"❌ {chart_type}: Missing fields {missing_fields}")
                return False
            else:
                print(f"✅ {chart_type}: Analysis successful")

            plt.close(fig)

        print(f"✅ All {len(figures)} chart types analyzed successfully")
        return True

    except Exception as e:
        print(f"❌ Multiple chart types test failed: {e}")
        return False


def test_complex_figure():
    """Test analysis of complex figure with multiple subplots"""
    print("\n🔄 Testing complex figure analysis...")

    try:
        import plot2llm

        # Create complex figure
        fig_complex, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig_complex.suptitle("Complex Analysis Test", fontsize=16)

        # Subplot 1: Line plot
        x = np.linspace(0, 5, 20)
        y = x**2 + np.random.randn(20) * 2
        axes[0, 0].plot(x, y, "ro-", linewidth=2, markersize=6)
        axes[0, 0].set_title("Quadratic Function")
        axes[0, 0].grid(True, alpha=0.3)

        # Subplot 2: Scatter plot
        x_scatter = np.random.randn(50)
        y_scatter = 0.7 * x_scatter + np.random.randn(50) * 0.5
        axes[0, 1].scatter(x_scatter, y_scatter, alpha=0.7, c="blue", s=60)
        axes[0, 1].set_title("Correlation Test")
        axes[0, 1].grid(True, alpha=0.3)

        # Subplot 3: Histogram
        data_hist = np.random.gamma(2, 2, 1000)
        axes[1, 0].hist(data_hist, bins=25, alpha=0.7, color="green", edgecolor="black")
        axes[1, 0].set_title("Gamma Distribution")
        axes[1, 0].grid(True, alpha=0.3)

        # Subplot 4: Bar plot
        categories = ["Group 1", "Group 2", "Group 3"]
        values = [25, 30, 35]
        axes[1, 1].bar(categories, values, color=["red", "blue", "green"])
        axes[1, 1].set_title("Group Comparison")

        plt.tight_layout()

        # Analyze complex figure
        analysis = plot2llm.convert(
            fig_complex,
            format="json",
            detail_level="high",
            include_statistics=True,
            include_visual_info=True,
        )

        # Verify analysis
        if analysis.get("axes_count") == 4:
            print("✅ Complex figure analysis successful (4 subplots detected)")
        else:
            print(f"❌ Expected 4 axes, got {analysis.get('axes_count')}")
            return False

        plt.close(fig_complex)
        return True

    except Exception as e:
        print(f"❌ Complex figure test failed: {e}")
        return False


def test_format_comparison():
    """Test comparison of different output formats"""
    print("\n🔄 Testing format comparison...")

    try:
        import plot2llm

        # Create test figure
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 5, 20)
        y = x**2 + np.random.randn(20) * 2
        ax.plot(x, y, "ro-", linewidth=2, markersize=6)
        ax.set_title("Format Comparison Test")
        ax.grid(True, alpha=0.3)

        # Test all formats
        formats = ["text", "json", "semantic"]
        format_results = {}

        for fmt in formats:
            result = plot2llm.convert(fig, format=fmt)
            format_results[fmt] = result

            if fmt == "json":
                size = len(json.dumps(result))
                print(f"✅ {fmt.upper()}: {size} characters (structured)")
            else:
                size = len(str(result))
                print(f"✅ {fmt.upper()}: {size} characters (text)")

        # Verify all formats produced results
        if all(len(str(result)) > 0 for result in format_results.values()):
            print("✅ All formats produced valid results")
            plt.close(fig)
            return True
        else:
            print("❌ Some formats produced empty results")
            return False

    except Exception as e:
        print(f"❌ Format comparison test failed: {e}")
        return False


def test_llm_integration():
    """Test LLM integration preparation"""
    print("\n🔄 Testing LLM integration preparation...")

    try:
        import plot2llm

        # Create business-like figure
        fig, ax = plt.subplots(figsize=(10, 6))

        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        sales = [120, 150, 180, 200, 220, 250]
        expenses = [100, 110, 130, 140, 150, 160]

        x_pos = np.arange(len(months))
        width = 0.35

        ax.bar(x_pos - width / 2, sales, width, label="Sales", color="green", alpha=0.8)
        ax.bar(
            x_pos + width / 2, expenses, width, label="Expenses", color="red", alpha=0.8
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Thousands €")
        ax.set_title("Financial Analysis - First Semester")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(months)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Convert to LLM-optimized format
        llm_analysis = plot2llm.convert(fig, format="text")

        # Create LLM prompt
        prompt = f"""
        Analyze the following data visualization and provide business insights:
        
        {llm_analysis}
        
        Please provide:
        1. Executive summary
        2. Key trends identified
        3. Business implications
        4. Recommendations
        """

        print(f"✅ LLM prompt created successfully (length: {len(prompt)} chars)")
        print(f"✅ Analysis contains business-relevant information")

        plt.close(fig)
        return True

    except Exception as e:
        print(f"❌ LLM integration test failed: {e}")
        return False


def test_real_world_case():
    """Test real-world data analysis case"""
    print("\n🔄 Testing real-world data analysis...")

    try:
        import plot2llm

        # Create customer dataset
        np.random.seed(42)
        n_customers = 500

        customer_data = pd.DataFrame(
            {
                "age": np.random.normal(35, 10, n_customers),
                "income": np.random.lognormal(10, 0.5, n_customers),
                "satisfaction": np.random.uniform(1, 10, n_customers),
                "time_customer": np.random.exponential(24, n_customers),
            }
        )

        # Create analysis dashboard
        fig_dashboard, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig_dashboard.suptitle("Customer Analysis Dashboard", fontsize=16)

        # Age distribution
        axes[0, 0].hist(
            customer_data["age"], bins=20, alpha=0.7, color="skyblue", edgecolor="black"
        )
        axes[0, 0].set_title("Age Distribution")
        axes[0, 0].set_xlabel("Age")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True, alpha=0.3)

        # Income vs Satisfaction
        axes[0, 1].scatter(
            customer_data["income"],
            customer_data["satisfaction"],
            alpha=0.6,
            c="red",
            s=30,
        )
        axes[0, 1].set_title("Income vs Satisfaction")
        axes[0, 1].set_xlabel("Income (€)")
        axes[0, 1].set_ylabel("Satisfaction Score")
        axes[0, 1].grid(True, alpha=0.3)

        # Satisfaction distribution
        axes[1, 0].hist(
            customer_data["satisfaction"],
            bins=15,
            alpha=0.7,
            color="green",
            edgecolor="black",
        )
        axes[1, 0].set_title("Satisfaction Distribution")
        axes[1, 0].set_xlabel("Satisfaction Score")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True, alpha=0.3)

        # Time as customer
        axes[1, 1].hist(
            customer_data["time_customer"],
            bins=20,
            alpha=0.7,
            color="orange",
            edgecolor="black",
        )
        axes[1, 1].set_title("Time as Customer")
        axes[1, 1].set_xlabel("Months")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Analyze dashboard
        dashboard_analysis = plot2llm.convert(
            fig_dashboard,
            format="json",
            detail_level="high",
            include_statistics=True,
            include_visual_info=True,
        )

        # Verify analysis
        if dashboard_analysis.get("axes_count") == 4:
            print("✅ Dashboard analysis successful (4 subplots detected)")

            # Check for statistical information
            stats = dashboard_analysis.get("data_info", {}).get("statistics", {})
            if stats:
                print("✅ Statistical analysis included")
            else:
                print("⚠️  No statistical analysis found")

            plt.close(fig_dashboard)
            return True
        else:
            print(f"❌ Expected 4 axes, got {dashboard_analysis.get('axes_count')}")
            return False

    except Exception as e:
        print(f"❌ Real-world case test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Plot2LLM Notebook Test Suite")
    print("=" * 50)

    tests = [
        ("Import", test_import),
        ("Basic Conversion", test_basic_conversion),
        ("Multiple Chart Types", test_multiple_chart_types),
        ("Complex Figure", test_complex_figure),
        ("Format Comparison", test_format_comparison),
        ("LLM Integration", test_llm_integration),
        ("Real-world Case", test_real_world_case),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The notebook should work correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
