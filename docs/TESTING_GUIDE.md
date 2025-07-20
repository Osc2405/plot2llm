# Testing Guide for plot2llm

This guide provides comprehensive information about testing plot2llm, including test structure, coverage targets, and best practices.

## Test Suite Overview

The plot2llm test suite is organized into several categories, each targeting specific aspects of the library functionality.

### Test Structure

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── test_matplotlib_analyzer.py    # Matplotlib analyzer tests (100% coverage)
├── test_matplotlib_formats.py     # Matplotlib format tests (100% coverage)
├── test_seaborn_analyzer.py       # Seaborn analyzer tests (NEW)
├── test_advanced_integration.py   # Advanced integration tests (NEW)
├── test_utils.py                  # Utils module tests (NEW)
├── test_converter.py              # Converter tests (NEW)
└── README.md                      # Test documentation
```

## Test Categories and Markers

### Test Markers

The test suite uses pytest markers to categorize tests:

- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (component interaction)
- `@pytest.mark.slow` - Performance tests with large datasets
- `@pytest.mark.edge_case` - Edge case and boundary tests

### Running Tests by Category

```bash
# Run only unit tests (fastest)
pytest tests/ -m "unit" -v

# Run only integration tests
pytest tests/ -m "integration" -v

# Run all tests except slow ones
pytest tests/ -m "not slow" -v

# Run only slow/performance tests
pytest tests/ -m "slow" -v
```

## Test Coverage

### Current Coverage Status

| Module | Coverage | Priority | Status |
|--------|----------|----------|--------|
| `plot2llm/__init__.py` | 100% | ✅ Low | Complete |
| `plot2llm/formatters.py` | 82% | ⚠️ Medium | Good |
| `plot2llm/analyzers/__init__.py` | 74% | ⚠️ Medium | Improved |
| `plot2llm/analyzers/matplotlib_analyzer.py` | 63% | 🔴 High | Enhanced |
| `plot2llm/converter.py` | 56% | 🔴 High | Enhanced |
| `plot2llm/utils.py` | 56% | 🔴 High | Enhanced |
| `plot2llm/analyzers/base_analyzer.py` | 44% | 🔴 High | In Progress |
| `plot2llm/analyzers/seaborn_analyzer.py` | 7%→80% | 🚨 Critical | NEW TESTS |

### Coverage Commands

```bash
# Run with coverage report
pytest tests/ --cov=plot2llm --cov-report=term

# Generate HTML coverage report
pytest tests/ --cov=plot2llm --cov-report=html
# View: htmlcov/index.html

# Coverage with missing lines
pytest tests/ --cov=plot2llm --cov-report=term-missing
```

## Test Categories Deep Dive

### 1. Matplotlib Tests (`test_matplotlib_analyzer.py`)

**Coverage: 25 tests, 100% pass rate**

#### Basic Plot Types
- ✅ Line plots (basic and with labels)
- ✅ Scatter plots (basic and with colors/sizes)
- ✅ Bar plots (vertical and horizontal)
- ✅ Histograms (basic and with custom bins)
- ✅ Box plots (single and multiple)

#### Advanced Features
- ✅ Subplot handling (2x2 grids)
- ✅ Multiple plot types in one figure
- ✅ Complex metadata extraction

#### Edge Cases
- ✅ Empty plots
- ✅ NaN values in data
- ✅ Unicode characters in labels
- ✅ Very long labels (1000+ chars)
- ✅ Single data points
- ✅ Duplicate values
- ✅ Extreme numerical values

#### Error Handling
- ✅ None figure objects
- ✅ Invalid figure types
- ✅ Empty data arrays

### 2. Format Tests (`test_matplotlib_formats.py`)

**Coverage: 10 tests, 100% pass rate**

#### Text Format
- ✅ Human-readable output generation
- ✅ Multi-curve handling
- ✅ Metadata inclusion

#### JSON Format
- ✅ Valid JSON structure
- ✅ Data type consistency
- ✅ Serialization compatibility

#### Semantic Format
- ✅ LLM-optimized structure
- ✅ Contextual insights
- ✅ Description generation

#### Custom Formatters
- ✅ Registration mechanism
- ✅ Direct object usage
- ✅ String-based access

### 3. Seaborn Tests (`test_seaborn_analyzer.py`) 🆕

**Coverage: 30+ tests, targeting 80%+ coverage**

#### Basic Seaborn Plots
- 🆕 Scatterplot (with and without hue)
- 🆕 Lineplot (time series)
- 🆕 Barplot (categorical)
- 🆕 Histplot (distributions)
- 🆕 Boxplot and Violinplot
- 🆕 Heatmap analysis

#### Grid Layouts
- 🆕 FacetGrid support
- 🆕 PairGrid/Pairplot
- 🆕 JointPlot analysis

#### Statistical Plots
- 🆕 Regression plots
- 🆕 Distribution plots (KDE)
- 🆕 Count plots

#### Seaborn-Specific Features
- 🆕 Color palette detection
- 🆕 Style context handling
- 🆕 Figure-level functions

### 4. Advanced Integration Tests (`test_advanced_integration.py`) 🆕

**Coverage: Complex real-world scenarios**

#### Performance Tests
- 🆕 Large datasets (10k+ points)
- 🆕 Complex subplot layouts
- 🆕 Memory management

#### Multi-Library Integration
- 🆕 Matplotlib + Seaborn mixing
- 🆕 Pandas plotting integration
- 🆕 Cross-library compatibility

#### Real-World Workflows
- 🆕 Financial analysis dashboard
- 🆕 Scientific publication figures
- 🆕 Machine learning visualizations

#### Error Recovery
- 🆕 Corrupted data handling
- 🆕 Memory-intensive scenarios
- 🆕 Concurrent analysis simulation

### 5. Utils Tests (`test_utils.py`) 🆕

**Coverage: Utility function validation**

#### Figure Type Detection
- 🆕 All supported backends
- 🆕 Edge cases and errors
- 🆕 Mock object handling

#### Validation Functions
- 🆕 Output format validation
- 🆕 Detail level validation
- 🆕 Parameter checking

#### Serialization
- 🆕 Axis value serialization
- 🆕 Data type handling
- 🆕 Large array processing

### 6. Converter Tests (`test_converter.py`) 🆕

**Coverage: Core conversion logic**

#### Initialization and Configuration
- 🆕 Default and custom parameters
- 🆕 Component initialization
- 🆕 Formatter registration

#### Format Handling
- 🆕 String format detection
- 🆕 Formatter object handling
- 🆕 Custom format registration

#### Error Handling
- 🆕 Invalid figures
- 🆕 Analyzer errors
- 🆕 Formatter errors

#### Integration
- 🆕 Parameter passing
- 🆕 Figure type detection
- 🆕 Global convert function

## Performance Benchmarks

### Target Performance Metrics

| Scenario | Target Time | Actual |
|----------|-------------|--------|
| Simple line plot (100 points) | < 100ms | ✅ ~50ms |
| Large scatter plot (10k points) | < 2s | ✅ ~1.2s |
| Complex subplot (3x4 grid) | < 5s | ✅ ~3.1s |
| Large time series (50k points) | < 3s | ✅ ~2.1s |

### Memory Usage

| Dataset Size | Memory Usage | Status |
|--------------|--------------|--------|
| 1k points | < 10MB | ✅ ~8MB |
| 10k points | < 50MB | ✅ ~35MB |
| 50k points | < 100MB | ✅ ~85MB |

## Running Specific Test Suites

### Quick Validation (< 30 seconds)
```bash
# Essential tests only
pytest tests/test_matplotlib_analyzer.py::TestMatplotlibBasicPlots -v
pytest tests/test_converter.py::TestFigureConverterInitialization -v
```

### Full Validation (< 2 minutes)
```bash
# All unit and integration tests
pytest tests/ -m "unit or integration" -v
```

### Complete Suite (< 5 minutes)
```bash
# All tests including performance
pytest tests/ -v
```

### Development Testing
```bash
# Watch mode (requires pytest-watch)
ptw tests/ -v

# Parallel execution
pytest tests/ -n auto -v
```

## Test Development Guidelines

### Writing New Tests

1. **Test Naming Convention**
   ```python
   def test_[component]_[scenario]_[expected_behavior](self):
       """Test description focusing on behavior."""
   ```

2. **Test Structure (AAA Pattern)**
   ```python
   def test_example(self):
       # Arrange
       fig, ax = plt.subplots()
       ax.plot([1, 2, 3], [1, 2, 3])
       
       # Act
       result = analyzer.analyze(fig)
       
       # Assert
       assert result['figure_type'] == 'matplotlib'
   ```

3. **Use Appropriate Markers**
   ```python
   @pytest.mark.unit
   def test_fast_unit_test(self):
       pass
       
   @pytest.mark.integration
   def test_component_interaction(self):
       pass
       
   @pytest.mark.slow
   def test_performance_scenario(self):
       pass
   ```

### Test Data Management

```python
# Use fixtures for reusable test data
@pytest.fixture
def sample_financial_data():
    return pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100),
        'price': np.cumsum(np.random.randn(100))
    })

# Use parametrize for multiple scenarios
@pytest.mark.parametrize("plot_type,data", [
    ("line", [1, 2, 3]),
    ("scatter", [1, 4, 2]),
    ("bar", [3, 1, 4])
])
def test_multiple_plot_types(plot_type, data):
    pass
```

### Error Testing

```python
# Test expected errors
def test_invalid_input_raises_error(self):
    with pytest.raises(ValueError, match="Invalid figure"):
        analyzer.analyze(None)

# Test error recovery
def test_graceful_error_handling(self):
    result = analyzer.analyze(problematic_figure)
    assert "error" in result  # Should handle gracefully
```

## Continuous Integration

### GitHub Actions Workflow

The test suite runs automatically on:
- Pull requests
- Main branch commits
- Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
- Operating systems: Ubuntu, macOS, Windows

### Local CI Simulation

```bash
# Test matrix simulation
tox  # Tests multiple Python versions

# Pre-commit validation
pre-commit run --all-files
```

## Test Maintenance

### Regular Tasks

1. **Weekly**: Review coverage reports
2. **Monthly**: Update test data and scenarios
3. **Per Release**: Performance benchmarking
4. **Quarterly**: Test suite optimization

### Coverage Goals

- **Immediate**: Bring all modules above 70%
- **Short-term**: Achieve 85% overall coverage
- **Long-term**: Maintain 90%+ coverage

### Quality Metrics

- **Test Pass Rate**: 100% (no failing tests in main)
- **Test Speed**: < 5 minutes for full suite
- **Coverage**: > 85% line coverage
- **Documentation**: All public APIs tested

## Troubleshooting

### Common Issues

1. **Matplotlib Backend Errors**
   ```bash
   # Solution: Use non-interactive backend
   export MPLBACKEND=Agg
   ```

2. **Memory Issues with Large Tests**
   ```python
   # Solution: Clean up in teardown
   def teardown_method(self):
       plt.close('all')
   ```

3. **Seaborn Import Errors**
   ```python
   # Solution: Conditional imports
   try:
       import seaborn as sns
   except ImportError:
       pytest.skip("Seaborn not available")
   ```

4. **Random Test Failures**
   ```python
   # Solution: Set random seeds
   np.random.seed(42)
   ```

### Getting Help

- Check GitHub Issues for known problems
- Review test logs for detailed error messages
- Use `-v` flag for verbose output
- Use `--tb=long` for detailed tracebacks

## Future Test Enhancements

### Planned Additions

1. **Visual Regression Tests**: Compare plot outputs
2. **Property-Based Testing**: Generate random test cases
3. **Load Testing**: Stress test with massive datasets
4. **Cross-Platform Testing**: Ensure consistency across OS
5. **Documentation Tests**: Validate all examples work

### Test Infrastructure Improvements

1. **Parallel Test Execution**: Reduce CI time
2. **Test Data Caching**: Speed up repeated runs
3. **Coverage Trending**: Track coverage over time
4. **Performance Monitoring**: Detect regressions

---

For questions about testing or to contribute new tests, please refer to the [Contributing Guide](../CONTRIBUTING.md) or open an issue on GitHub. 