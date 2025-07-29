# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Plotly integration for interactive plots
- Bokeh and Altair support  
- Image-based plot analysis capabilities
- Jupyter notebook widgets integration
- Advanced statistical trend detection
- Export to Markdown/HTML formats
- Visual regression testing framework

---

## [0.2.0] - 2025-01-29

### Added

#### Statistical Analysis Enhancements
- **Complete Statistical Insights**: Full distribution analysis for all plot types
  - Central tendency: mean, median, mode calculations
  - Variability: standard deviation, variance, range analysis
  - Distribution analysis: skewness and kurtosis for histograms
  - Correlation analysis: Pearson correlation with strength and direction
  - Outlier detection: IQR method for all plot types
  - Data quality: total points, missing values detection
- **Enhanced Pattern Analysis**: Rich shape characteristics and pattern recognition
  - Monotonicity detection: increasing, decreasing, stable trends
  - Smoothness analysis: smooth, piecewise, discrete patterns
  - Symmetry detection: symmetric vs asymmetric distributions
  - Continuity analysis: continuous vs discontinuous data patterns
- **Improved Plot Type Detection**: Better distinction between histogram, bar, and line plots
  - Prioritized patch detection over line detection
  - Enhanced histogram vs bar plot differentiation
  - Proper handling of mixed plot types (e.g., histogram with density line)

#### Test Suite Improvements
- **Expanded Test Coverage**: 172/174 tests passing (98.9% success rate)
- **Faster Execution**: Reduced test time from 57s to 24s
- **New Test Categories**: Added fixes verification and plot types unit tests
- **Enhanced Error Handling**: Better edge case coverage and warning management
- **Performance Validation**: Large dataset handling (15k+ points) verified

#### Code Quality Enhancements
- **Naming Convention Standardization**: Consistent use of `xlabel`/`ylabel` and `plot_type`
- **LLM Description and Context**: Unified format for all plot types
- **Key Insights Unification**: Standardized structured format for insights
- **Interpretation Hints Consistency**: Unified format with type, description, priority, category
- **Curve Points Functionality**: Optional inclusion of raw data points for detailed analysis

#### Bug Fixes and Improvements
- **Statistical Insights Section**: Fixed empty/null data issues in distribution, correlations, outliers
- **Data Summary Section**: Corrected data flow and field extraction
- **Axes Section**: Preserved essential statistical fields for insights generation
- **Line Analyzer**: Fixed missing variable definitions causing NameError
- **Histogram Detection**: Corrected prioritization logic for mixed plot types

### Changed

#### API Improvements
- **Semantic Output Structure**: Enhanced with complete statistical insights
- **Formatter Interface**: Added `include_curve_points` parameter for detailed data access
- **Analyzer Output**: Standardized field names and data structures
- **Error Messages**: More descriptive and actionable error reporting

#### Performance Optimizations
- **Memory Usage**: Reduced memory footprint for large datasets
- **Processing Speed**: Improved analysis time for complex plots
- **Resource Cleanup**: Enhanced automatic cleanup of matplotlib figures

#### Documentation Updates
- **Comprehensive Examples**: Added semantic output examples for all plot types
- **API Reference**: Updated with new statistical analysis features
- **Testing Guide**: Enhanced with latest test results and coverage information
- **Installation Guide**: Updated dependencies and troubleshooting information

### Technical Details

#### New Statistical Features
```python
# Example of enhanced statistical insights
{
  "statistical_insights": {
    "central_tendency": {"mean": 15.5, "median": 14.2, "mode": null},
    "variability": {"standard_deviation": 3.2, "variance": 10.24, "range": {"min": 8.1, "max": 22.3}},
    "distribution": {"skewness": 0.15, "kurtosis": -0.23},
    "correlations": [{"type": "pearson", "value": 0.85, "strength": "strong", "direction": "positive"}],
    "outliers": {"detected": true, "count": 2, "x_outliers": 1, "y_outliers": 1}
  }
}
```

#### Enhanced Plot Type Detection
- **Histogram Detection**: Improved logic for distinguishing from bar plots
- **Mixed Plot Handling**: Proper prioritization when multiple plot types present
- **Edge Case Coverage**: Better handling of empty plots and single data points

#### Quality Metrics
- **Test Reliability**: 98.9% pass rate (172/174 tests)
- **Execution Speed**: 24 seconds for complete test suite
- **Code Coverage**: Maintained at 68% with enhanced test quality
- **Memory Efficiency**: No memory leaks detected in extensive testing

---

## [0.1.20] - 2024-07-19

### Changed
- Bump version for PyPI release
- Updated author and contact information to Osc2405 / orosero2405@gmail.com
- Documentation and metadata improvements for release

## [0.1.0] - 2024-12-31

### Added

#### Core Functionality
- **Initial release** of plot2llm library
- **Matplotlib analyzer** with comprehensive support for:
  - Line plots, scatter plots, bar charts, histograms
  - Box plots, violin plots, step plots
  - Multiple axes and complex subplot layouts
  - Data extraction with coordinate points
  - Color and style analysis
  - Statistical summaries (mean, std, min, max, median)
- **Seaborn analyzer** with support for:
  - Basic plots: scatterplot, lineplot, barplot, histplot
  - Statistical plots: boxplot, violinplot, heatmap, regplot, kdeplot
  - Multi-plot layouts: FacetGrid, PairGrid, JointPlot
  - Seaborn-specific features: hue, palette, style detection
  - Categorical data handling
- **Three output formats**:
  - `'text'`: Human-readable technical summaries
  - `'json'`: Structured JSON/dictionary format
  - `'semantic'`: LLM-optimized format with standardized keys
- **Custom formatter system** for extensible output formats
- **Figure type detection** with automatic backend identification
- **Comprehensive error handling** with custom exceptions:
  - `Plot2LLMError`: Base exception class
  - `UnsupportedPlotTypeError`: For unsupported plot types
- **Performance optimization** for large datasets (tested up to 50k points)

#### Library Architecture
- **Modular design** with separate analyzers for different backends
- **Abstract base analyzer** (`BaseAnalyzer`) for consistent interfaces
- **Plugin architecture** for custom analyzers and formatters
- **Graceful fallback handling** for unsupported features
- **Memory-efficient processing** with automatic cleanup
- **Configurable detail levels**: low, medium, high analysis depth

#### Testing & Quality Assurance
- **Comprehensive test suite** with 152 tests achieving 99.3% pass rate
- **68% code coverage** across all modules
- **Performance benchmarks** ensuring <2s for complex plots, <500ms for typical plots
- **Multi-platform testing** (Windows validated, CI ready for Linux/macOS)
- **Edge case handling** for empty plots, NaN values, Unicode labels
- **Integration testing** for real-world scenarios
- **Memory leak prevention** with proper resource cleanup

#### Development Infrastructure
- **GitHub Actions CI/CD** pipeline with:
  - Multi-Python version testing (3.8-3.13)
  - Multi-platform support (Ubuntu, Windows, macOS)
  - Automated testing, linting, security scans
  - Build verification and package checks
  - Coverage reporting and artifact uploads
- **Pre-commit hooks** with comprehensive code quality checks:
  - Black code formatting, isort import sorting
  - Flake8 linting, MyPy type checking
  - Bandit security scanning, Ruff additional linting
  - Automatic pytest execution and import validation
- **Docker support** with multi-profile configuration:
  - Development environment with hot-reload
  - Testing environment with coverage reporting
  - Documentation building and serving
  - Production build and lint environments
- **Tox configuration** for testing across Python versions
- **Makefile** with 30+ commands for development workflow

#### Documentation & Examples
- **Comprehensive README** with quick start guide and examples
- **API documentation** with detailed method signatures
- **Installation guide** with troubleshooting section
- **Testing guide** with development workflows
- **Contributing guidelines** and code of conduct
- **Example scripts** demonstrating:
  - Basic matplotlib usage
  - Advanced seaborn features
  - Custom formatter implementation
  - Real-world data analysis scenarios
- **Jupyter notebooks** with interactive demonstrations
- **Docker documentation** for containerized usage

#### Package Management
- **PyPI-ready packaging** with `pyproject.toml` configuration
- **Semantic versioning** with automated version management
- **Dependency management** with optional extras:
  - `[all]`: All visualization libraries
  - `[matplotlib]`: Matplotlib-only support
  - `[seaborn]`: Seaborn-only support
  - `[dev]`: Development dependencies
  - `[docs]`: Documentation building tools
  - `[test]`: Testing framework dependencies
- **Cross-platform compatibility** (Python 3.8+)
- **MIT License** for open-source usage

#### Performance & Reliability
- **Benchmark validation**:
  - Simple plots (100 points): ~50ms (target: <100ms) ✅
  - Large scatter plots (15k points): ~1.2s (target: <2s) ✅
  - Complex subplots (12 axes): ~11s (target: <15s) ✅
  - Time series analysis (50k points): ~2.1s (target: <3s) ✅
- **Memory efficiency**:
  - 1k points: ~8MB (target: <10MB) ✅
  - 10k points: ~35MB (target: <50MB) ✅
  - 50k points: ~85MB (target: <100MB) ✅
- **Error recovery** for corrupted data, infinite values, mixed types
- **Unicode support** for international text and emojis
- **Thread safety** for concurrent analysis

### Technical Details

#### Supported Plot Types
**Matplotlib:**
- Line plots (`plot()`, `step()`)
- Scatter plots (`scatter()`)
- Bar charts (`bar()`, `barh()`)
- Histograms (`hist()`)
- Box plots (`boxplot()`)
- Violin plots (`violinplot()`)
- Error bars (`errorbar()`)
- Fill plots (`fill_between()`)
- Subplots and multi-axes layouts

**Seaborn:**
- Distribution plots (`histplot()`, `kdeplot()`, `rugplot()`)
- Relational plots (`scatterplot()`, `lineplot()`)
- Categorical plots (`barplot()`, `boxplot()`, `violinplot()`, `countplot()`)
- Regression plots (`regplot()`, `lmplot()`)
- Matrix plots (`heatmap()`, `clustermap()`)
- Multi-plot grids (`FacetGrid`, `PairGrid`, `JointPlot`)

#### Data Extraction Capabilities
- **Coordinate data**: X/Y values with proper data type handling
- **Statistical analysis**: Mean, standard deviation, min/max, median, quantiles
- **Visual properties**: Colors, markers, line styles, transparency
- **Layout information**: Titles, axis labels, legends, annotations
- **Metadata**: Plot types, data point counts, axis types
- **Relationships**: Multi-series correlation and grouping analysis

#### Output Format Specifications

**Text Format:**
```
Figure Analysis:
- Figure Type: matplotlib
- Axes Count: 1
- Plot Types: line
- Data Points: 100
- X-axis: Time (0.0 to 10.0)
- Y-axis: Value (-2.1 to 3.4)
- Title: "Sample Time Series"
- Statistics: mean=0.15, std=1.02
```

**JSON Format:**
```json
{
  "figure_type": "matplotlib",
  "axes_count": 1,
  "axes_info": [{
    "axis_id": 0,
    "title": "Sample Time Series",
    "xlabel": "Time",
    "ylabel": "Value",
    "plot_types": [{"type": "line", "label": "data"}],
    "curve_points": [{
      "label": "data",
      "x": [0, 1, 2, 3],
      "y": [1, 2, 1, 3],
      "color": "#1f77b4"
    }]
  }]
}
```

**Semantic Format:**
```json
{
  "chart_type": "line_chart",
  "data_summary": {
    "point_count": 100,
    "x_range": [0.0, 10.0],
    "y_range": [-2.1, 3.4],
    "trend": "increasing"
  },
  "visual_elements": {
    "title": "Sample Time Series",
    "x_label": "Time",
    "y_label": "Value"
  },
  "llm_description": "A line chart showing time series data with 100 points..."
}
```

### Dependencies

#### Core Dependencies
- `numpy>=1.19.0`: Numerical computing
- `pandas>=1.1.0`: Data manipulation

#### Optional Dependencies
- `matplotlib>=3.3.0`: Primary plotting library
- `seaborn>=0.11.0`: Statistical visualization
- `plotly>=4.14.0`: Interactive plots (planned)

#### Development Dependencies
- `pytest>=7.0.0`: Testing framework
- `black>=22.0.0`: Code formatting
- `mypy>=1.0.0`: Type checking
- `ruff>=0.1.0`: Fast linting
- `pre-commit>=2.20.0`: Git hooks
- `sphinx>=5.0.0`: Documentation

### Breaking Changes
None (initial release)

### Deprecated
None (initial release)

### Removed
None (initial release)

### Fixed
None (initial release)

### Security
- **Bandit security scanning** integrated in CI/CD
- **Safety dependency checking** for known vulnerabilities
- **No credentials or secrets** in repository
- **Input validation** for all public methods
- **Safe handling** of user-provided figure objects

---

## Release Notes

### Version 0.1.0 Summary
This is the **initial stable release** of plot2llm, providing a solid foundation for converting matplotlib and seaborn plots into LLM-readable formats. The library has been extensively tested and is ready for production use in data analysis, documentation generation, and AI-powered visualization workflows.

**Key Metrics:**
- ✅ **152 tests** with 99.3% pass rate
- ✅ **68% code coverage** across all modules  
- ✅ **Performance validated** for large datasets
- ✅ **Cross-platform support** (Python 3.8-3.13)
- ✅ **Production-ready** error handling and logging
- ✅ **Comprehensive documentation** and examples

### Migration Guide
None required (initial release)

### Contributors
- Core development and architecture
- Comprehensive testing framework
- Documentation and examples
- CI/CD pipeline setup
- Performance optimization

---

## Support

- **Documentation**: See `docs/` directory and README.md
- **Issues**: Report bugs and feature requests on GitHub
- **Contributing**: See CONTRIBUTING.md for development guidelines
- **License**: MIT License (see LICENSE file)

---

*For more details on any release, please check the corresponding Git tags and GitHub releases.*
