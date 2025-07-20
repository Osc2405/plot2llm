# Contributing to plot2llm

We love your input! We want to make contributing to plot2llm as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. **Fork the repo** and create your branch from `main`.
2. **Make your changes** with clear, concise commits.
3. **Add tests** if you've added code that should be tested.
4. **Update documentation** if you've changed APIs.
5. **Ensure the test suite passes** locally.
6. **Make sure your code lints** with our style guidelines.
7. **Submit your pull request!**

## Setting Up Development Environment

### Prerequisites

- Python 3.8 or higher
- Git

### Quick Setup

```bash
# 1. Clone your fork
git clone https://github.com/your-username/plot2llm.git
cd plot2llm

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in development mode
pip install -e ".[dev,all]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Run tests to verify setup
python -m pytest tests/ -v
```

### Development Dependencies

Install all development dependencies:

```bash
pip install -e ".[dev,all]"
```

This includes:
- Testing: `pytest`, `pytest-cov`, `pytest-mock`
- Code quality: `black`, `ruff`, `mypy`, `isort`
- Documentation: `sphinx`, `sphinx-rtd-theme`
- Pre-commit hooks: `pre-commit`

## Testing

We use `pytest` for our test suite. All contributions should include appropriate tests.

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/ -m "unit" -v          # Unit tests only
python -m pytest tests/ -m "integration" -v   # Integration tests only

# Run with coverage
python -m pytest tests/ --cov=plot2llm --cov-report=html

# Run specific test file
python -m pytest tests/test_matplotlib_analyzer.py -v
```

### Test Categories

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **Performance tests**: Test with large datasets (marked as `slow`)

### Writing Tests

1. **Place tests** in the appropriate `tests/test_*.py` file
2. **Use descriptive names**: `test_matplotlib_line_plot_basic`
3. **Follow AAA pattern**: Arrange, Act, Assert
4. **Use pytest fixtures** for reusable test data
5. **Add appropriate markers**: `@pytest.mark.unit`, `@pytest.mark.integration`

Example test:

```python
import pytest
import matplotlib.pyplot as plt
from plot2llm import convert

@pytest.mark.unit
def test_convert_simple_line_plot():
    """Test conversion of a simple line plot."""
    # Arrange
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 2])
    ax.set_title("Test Plot")
    
    # Act
    result = convert(fig, format='text')
    
    # Assert
    assert isinstance(result, str)
    assert "line" in result.lower()
    assert "Test Plot" in result
    
    # Cleanup
    plt.close(fig)
```

## Code Style

We use several tools to maintain code quality:

### Formatting and Linting

```bash
# Format code
black plot2llm/ tests/
isort plot2llm/ tests/

# Lint code
ruff check plot2llm/ tests/
flake8 plot2llm/ tests/
mypy plot2llm/

# Run all quality checks
make lint  # or python -m pytest --flake8 --mypy
```

### Style Guidelines

- **Line length**: 88 characters (Black default)
- **Import sorting**: Use `isort` with Black profile
- **Type hints**: Required for all public functions
- **Docstrings**: Google/NumPy style for all public classes and functions
- **Variable names**: Descriptive and snake_case

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install hooks (done once)
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Reporting Bugs

We use GitHub Issues to track public bugs. Report a bug by opening a new issue.

### Bug Report Template

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

### Example Bug Report

```
**Bug**: matplotlib subplot conversion fails with IndexError

**Environment**:
- plot2llm version: 0.1.19
- Python version: 3.9.7
- OS: Windows 10

**Steps to reproduce**:
1. Create subplot with `fig, axes = plt.subplots(2, 2)`
2. Call `convert(fig, 'json')`
3. IndexError occurs

**Expected**: Should return JSON with 4 axes
**Actual**: IndexError: list index out of range

**Sample code**:
```python
import matplotlib.pyplot as plt
import plot2llm

fig, axes = plt.subplots(2, 2)
result = plot2llm.convert(fig, 'json')  # Fails here
```

## Feature Requests

We welcome feature requests! Please:

1. **Check existing issues** first to avoid duplicates
2. **Explain the motivation** - what problem does it solve?
3. **Describe the solution** you'd like
4. **Consider alternatives** you've thought of
5. **Provide examples** if possible

## Documentation

Documentation improvements are always welcome!

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# View documentation
open _build/html/index.html  # On macOS
# Or navigate to docs/_build/html/index.html
```

### Documentation Guidelines

- Use clear, concise language
- Include code examples
- Update docstrings when changing APIs
- Add new features to appropriate documentation sections

## Community Guidelines

### Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Documentation**: Check our comprehensive docs first

## Release Process

For maintainers:

1. **Update version** in `plot2llm/__init__.py`, `setup.py`, and `pyproject.toml`
2. **Update CHANGELOG.md** with new features and fixes
3. **Run full test suite**: `python -m pytest tests/`
4. **Build and test package**: `python -m build && twine check dist/*`
5. **Test on TestPyPI**: `twine upload --repository testpypi dist/*`
6. **Create GitHub release** with tag
7. **Publish to PyPI**: `twine upload dist/*`

## Performance Considerations

When contributing:

- **Test with large datasets** when relevant
- **Consider memory usage** for large plots
- **Use appropriate data structures** (avoid unnecessary copies)
- **Profile performance-critical code**

## Supported Platforms

We support:
- **Python**: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13
- **Operating Systems**: Windows, macOS, Linux
- **Visualization Libraries**: matplotlib 3.3+, seaborn 0.11+

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## Recognition

Contributors are recognized in:
- GitHub contributor graphs
- CHANGELOG.md for significant contributions
- README.md for major features

---

## Quick Reference

### Common Commands

```bash
# Setup
pip install -e ".[dev,all]"
pre-commit install

# Testing
python -m pytest tests/ -v
python -m pytest tests/ --cov=plot2llm

# Code Quality
black plot2llm/ tests/
ruff check plot2llm/ tests/
mypy plot2llm/

# Build
python -m build
twine check dist/*
```

### File Structure

```
plot2llm/
├── plot2llm/           # Main package
│   ├── analyzers/      # Figure analyzers
│   ├── formatters.py   # Output formatters
│   ├── converter.py    # Main converter
│   └── utils.py        # Utilities
├── tests/              # Test suite
├── docs/               # Documentation
├── examples/           # Example scripts
└── notebooks/          # Jupyter notebooks
```

---

Thank you for contributing to plot2llm! 🎉 