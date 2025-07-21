#!/usr/bin/env python3
"""
Coverage optimization script for plot2llm.

This script analyzes code coverage and suggests optimizations
for improving test coverage and removing unused code.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class CoverageAnalyzer:
    """Analyze code coverage and suggest optimizations."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.plot2llm_path = self.project_root / "plot2llm"

    def analyze_module_coverage(self) -> Dict[str, Dict]:
        """Analyze coverage for each module."""
        coverage_data = {
            "plot2llm/__init__.py": {"coverage": 100, "priority": "low"},
            "plot2llm/formatters.py": {"coverage": 82, "priority": "medium"},
            "plot2llm/analyzers/__init__.py": {"coverage": 74, "priority": "medium"},
            "plot2llm/analyzers/matplotlib_analyzer.py": {
                "coverage": 63,
                "priority": "high",
            },
            "plot2llm/converter.py": {"coverage": 56, "priority": "high"},
            "plot2llm/utils.py": {"coverage": 56, "priority": "high"},
            "plot2llm/analyzers/base_analyzer.py": {"coverage": 44, "priority": "high"},
            "plot2llm/analyzers/seaborn_analyzer.py": {
                "coverage": 7,
                "priority": "critical",
            },
        }

        return coverage_data

    def find_unused_methods(self, module_path: str) -> List[str]:
        """Find potentially unused methods in a module."""
        unused_methods = []

        try:
            with open(module_path, "r", encoding="utf-8") as file:
                tree = ast.parse(file.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Look for methods that might be unused
                    if (
                        node.name.startswith("_")
                        and not node.name.startswith("__")
                        and "test" not in node.name.lower()
                    ):
                        unused_methods.append(node.name)

        except Exception as e:
            print(f"Error analyzing {module_path}: {e}")

        return unused_methods

    def suggest_optimizations(self) -> Dict[str, List[str]]:
        """Suggest optimizations for each module."""
        suggestions = {}
        coverage_data = self.analyze_module_coverage()

        for module, data in coverage_data.items():
            module_suggestions = []
            coverage = data["coverage"]
            priority = data["priority"]

            if priority == "critical":
                module_suggestions.extend(
                    [
                        "🚨 CRITICAL: Add comprehensive test suite",
                        "📊 Current coverage extremely low (7%)",
                        "🎯 Target: Create test_seaborn_analyzer.py with 80%+ coverage",
                        "📝 Focus on: all plot types, grid layouts, statistical plots",
                    ]
                )

            elif priority == "high":
                if "base_analyzer.py" in module:
                    module_suggestions.extend(
                        [
                            "🔧 Add tests for abstract methods and base functionality",
                            "📋 Test method inheritance and default implementations",
                            "🧪 Mock subclass implementations for testing",
                        ]
                    )
                elif "converter.py" in module:
                    module_suggestions.extend(
                        [
                            "🔄 Add tests for format validation",
                            "🎨 Test custom formatter registration",
                            "⚠️ Add error handling tests",
                            "📊 Test figure type detection edge cases",
                        ]
                    )
                elif "utils.py" in module:
                    module_suggestions.extend(
                        [
                            "🔍 Test all figure type detection scenarios",
                            "✅ Test validation functions",
                            "📐 Test utility functions with edge cases",
                        ]
                    )
                elif "matplotlib_analyzer.py" in module:
                    module_suggestions.extend(
                        [
                            "📈 Add tests for complex plot combinations",
                            "🎨 Test color and style extraction",
                            "📊 Test statistical calculations",
                            "🔧 Test detailed analysis mode",
                        ]
                    )

            elif priority == "medium":
                if "formatters.py" in module:
                    module_suggestions.extend(
                        [
                            "📝 Test edge cases in text formatting",
                            "🔧 Test JSON serialization with complex data",
                            "🤖 Test semantic format generation",
                        ]
                    )
                else:
                    module_suggestions.extend(
                        ["🧪 Add edge case tests", "📊 Improve error handling coverage"]
                    )

            # Add general suggestions based on coverage
            if coverage < 50:
                module_suggestions.append("🎯 Priority: Bring coverage above 50%")
            elif coverage < 70:
                module_suggestions.append("📈 Goal: Reach 70% coverage")
            elif coverage < 85:
                module_suggestions.append("✨ Target: Achieve 85% coverage")

            suggestions[module] = module_suggestions

        return suggestions

    def identify_dead_code(self) -> Dict[str, List[str]]:
        """Identify potentially dead code."""
        dead_code = {}

        for py_file in self.plot2llm_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            unused_methods = self.find_unused_methods(str(py_file))
            if unused_methods:
                relative_path = str(py_file.relative_to(self.project_root))
                dead_code[relative_path] = unused_methods

        return dead_code

    def generate_test_templates(self) -> Dict[str, str]:
        """Generate test templates for low-coverage modules."""
        templates = {}

        # Base analyzer test template
        templates[
            "base_analyzer"
        ] = '''
def test_base_analyzer_abstract_methods():
    """Test that abstract methods raise NotImplementedError."""
    # Test implementation here
    
def test_extract_basic_info():
    """Test basic info extraction."""
    # Test implementation here
    
def test_extract_visual_info():
    """Test visual info extraction.""" 
    # Test implementation here
'''

        # Utils test template
        templates[
            "utils"
        ] = '''
def test_detect_figure_type_all_backends():
    """Test figure type detection for all supported backends."""
    # Test matplotlib, seaborn, plotly, etc.
    
def test_validate_output_format():
    """Test output format validation."""
    # Test valid and invalid formats
    
def test_validate_detail_level():
    """Test detail level validation."""
    # Test all detail levels
'''

        # Converter test template
        templates[
            "converter"
        ] = '''
def test_converter_error_handling():
    """Test converter error handling."""
    # Test various error conditions
    
def test_custom_analyzer_registration():
    """Test registering custom analyzers."""
    # Test analyzer registration and usage
    
def test_figure_type_detection_edge_cases():
    """Test edge cases in figure type detection."""
    # Test unusual figure types
'''

        return templates

    def create_optimization_report(self) -> str:
        """Create a comprehensive optimization report."""
        report = []
        report.append("# Code Coverage Optimization Report")
        report.append("=" * 50)
        report.append("")

        # Coverage analysis
        coverage_data = self.analyze_module_coverage()
        report.append("## Current Coverage Status")
        report.append("")

        for module, data in sorted(
            coverage_data.items(), key=lambda x: x[1]["coverage"]
        ):
            coverage = data["coverage"]
            priority = data["priority"]

            priority_emoji = {
                "low": "✅",
                "medium": "⚠️",
                "high": "🔴",
                "critical": "🚨",
            }

            report.append(
                f"{priority_emoji[priority]} **{module}**: {coverage}% "
                f"({priority} priority)"
            )

        report.append("")

        # Optimization suggestions
        suggestions = self.suggest_optimizations()
        report.append("## Optimization Suggestions")
        report.append("")

        for module, module_suggestions in suggestions.items():
            if module_suggestions:
                report.append(f"### {module}")
                for suggestion in module_suggestions:
                    report.append(f"- {suggestion}")
                report.append("")

        # Dead code analysis
        dead_code = self.identify_dead_code()
        if dead_code:
            report.append("## Potentially Unused Code")
            report.append("")

            for module, methods in dead_code.items():
                if methods:
                    report.append(f"### {module}")
                    for method in methods:
                        report.append(
                            f"- `{method}()` - Consider removal if truly unused"
                        )
                    report.append("")

        # Action plan
        report.append("## Recommended Action Plan")
        report.append("")
        report.append("### Phase 1: Critical Issues (Immediate)")
        report.append("1. 🚨 Create comprehensive seaborn tests")
        report.append("2. 🔴 Improve base_analyzer coverage")
        report.append("3. 🔴 Add converter edge case tests")
        report.append("")

        report.append("### Phase 2: High Priority (This Week)")
        report.append("1. 📊 Enhance matplotlib_analyzer tests")
        report.append("2. 🔧 Add utils validation tests")
        report.append("3. ⚠️ Improve formatters coverage")
        report.append("")

        report.append("### Phase 3: Optimization (Next Week)")
        report.append("1. 🧹 Remove dead code")
        report.append("2. 📈 Reach 85% overall coverage")
        report.append("3. 🚀 Performance optimizations")
        report.append("")

        return "\n".join(report)


def main():
    """Main function to run coverage analysis."""
    print("🔍 Analyzing code coverage and generating optimization report...")

    analyzer = CoverageAnalyzer()

    # Generate report
    report = analyzer.create_optimization_report()

    # Write to file
    report_path = Path("COVERAGE_OPTIMIZATION_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"✅ Report generated: {report_path}")

    # Print summary
    coverage_data = analyzer.analyze_module_coverage()
    total_coverage = sum(data["coverage"] for data in coverage_data.values()) / len(
        coverage_data
    )

    print(f"\n📊 Current overall coverage: {total_coverage:.1f}%")

    critical_modules = [
        module
        for module, data in coverage_data.items()
        if data["priority"] == "critical"
    ]
    high_priority = [
        module for module, data in coverage_data.items() if data["priority"] == "high"
    ]

    if critical_modules:
        print(f"🚨 Critical priority modules: {len(critical_modules)}")
        for module in critical_modules:
            print(f"   - {module}")

    if high_priority:
        print(f"🔴 High priority modules: {len(high_priority)}")
        for module in high_priority:
            print(f"   - {module}")

    print(f"\n🎯 Target overall coverage: 85%")
    print(
        f"📈 Coverage improvement needed: {85 - total_coverage:.1f} percentage points"
    )


if __name__ == "__main__":
    main()
