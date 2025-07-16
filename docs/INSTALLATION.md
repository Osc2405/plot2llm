# Installation Guide - Plot2LLM

## Table of Contents

1. [Basic Installation](#basic-installation)
2. [Installation from PyPI](#installation-from-pypi)
3. [Installation from Source](#installation-from-source)
4. [Installation in Virtual Environments](#installation-in-virtual-environments)
5. [Dependencies](#dependencies)
6. [Installation Verification](#installation-verification)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Configuration](#advanced-configuration)

---

## Basic Installation

### System Requirements

- **Python**: 3.8 or higher
- **pip**: Version 19.0 or higher
- **Operating System**: Windows, macOS, Linux

### Verify Python Version

```bash
python --version
# Should show Python 3.8 or higher
```

### Verify pip

```bash
pip --version
# Should show pip 19.0 or higher
```

---

## Installation from PyPI

### Method 1: Simple Installation

```bash
pip install plot2llm
```

### Method 2: Installation with Specific Version

```bash
pip install plot2llm==0.1.5
```

### Method 3: Installation with Update

```bash
pip install --upgrade plot2llm
```

### Method 4: Installation with Optional Dependencies

```bash
# Install with all optional dependencies
pip install plot2llm[all]

# Or install specific dependencies
pip install plot2llm matplotlib seaborn plotly
```

---

## Installation from Source

### Method 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/your-username/plot2llm.git
cd plot2llm

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

### Method 2: Download and Install

```bash
# Download the .tar.gz file
wget https://github.com/your-username/plot2llm/archive/refs/tags/v0.1.5.tar.gz

# Extract
tar -xzf v0.1.5.tar.gz
cd plot2llm-0.1.5

# Install
pip install .
```

### Method 3: Installation from Wheel File

```bash
# Download the wheel
wget https://files.pythonhosted.org/packages/.../plot2llm-0.1.5-py3-none-any.whl

# Install
pip install plot2llm-0.1.5-py3-none-any.whl
```

---

## Installation in Virtual Environments

### Method 1: venv (Recommended)

```bash
# Create virtual environment
python -m venv plot2llm_env

# Activate virtual environment
# On Windows:
plot2llm_env\Scripts\activate

# On macOS/Linux:
source plot2llm_env/bin/activate

# Install plot2llm
pip install plot2llm
```

### Method 2: conda

```bash
# Create conda environment
conda create -n plot2llm_env python=3.9

# Activate environment
conda activate plot2llm_env

# Install plot2llm
pip install plot2llm
```

### Method 3: virtualenv

```bash
# Install virtualenv if not installed
pip install virtualenv

# Create virtual environment
virtualenv plot2llm_env

# Activate virtual environment
# On Windows:
plot2llm_env\Scripts\activate

# On macOS/Linux:
source plot2llm_env/bin/activate

# Install plot2llm
pip install plot2llm
```

---

## Dependencies

### Main Dependencies (Required)

- **numpy**: >= 1.19.0
- **pandas**: >= 1.1.0

### Optional Dependencies (Recommended)

- **matplotlib**: >= 3.3.0
- **seaborn**: >= 0.11.0
- **plotly**: >= 4.14.0

### Development Dependencies

- **pytest**: >= 6.0.0
- **build**: >= 0.7.0
- **twine**: >= 3.0.0

### Installing Dependencies

```bash
# Install main dependencies
pip install numpy pandas

# Install optional dependencies
pip install matplotlib seaborn plotly

# Install development dependencies
pip install pytest build twine
```

### Verifying Dependencies

```python
import plot2llm

# Verify that plot2llm can be imported
print("Plot2LLM installed successfully")

# Verify optional dependencies
try:
    import matplotlib
    print("Matplotlib available")
except ImportError:
    print("Matplotlib not available")

try:
    import seaborn
    print("Seaborn available")
except ImportError:
    print("Seaborn not available")

try:
    import plotly
    print("Plotly available")
except ImportError:
    print("Plotly not available")
```

---

## Installation Verification

### Method 1: Basic Verification

```python
import plot2llm

# Verify version
print(f"Plot2LLM version: {plot2llm.__version__}")

# Verify that main function works
print("Convert function available:", hasattr(plot2llm, 'convert'))
```

### Method 2: Functionality Test

```python
import matplotlib.pyplot as plt
import plot2llm

# Create simple figure
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 2])
ax.set_title('Installation Test')

# Test conversion
try:
    result = plot2llm.convert(fig, format='text')
    print("✅ Conversion successful")
    print("Result:", result[:100] + "..." if len(result) > 100 else result)
except Exception as e:
    print("❌ Conversion error:", e)

plt.close()
```

### Method 3: Run Tests

```bash
# Run basic tests
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_basic.py -v
```

---

## Troubleshooting

### Problem 1: Import Error

**Error:**
```
ModuleNotFoundError: No module named 'plot2llm'
```

**Solution:**
```bash
# Verify it's installed
pip list | grep plot2llm

# Reinstall if necessary
pip uninstall plot2llm
pip install plot2llm
```

### Problem 2: Dependency Error

**Error:**
```
ImportError: No module named 'numpy'
```

**Solution:**
```bash
# Install missing dependencies
pip install numpy pandas

# Or install all dependencies
pip install plot2llm[all]
```

### Problem 3: Permission Error

**Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Solution:**
```bash
# Use --user flag
pip install --user plot2llm

# Or use sudo (Linux/macOS)
sudo pip install plot2llm

# Or use virtual environment (recommended)
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows
pip install plot2llm
```

### Problem 4: Python Version Error

**Error:**
```
SyntaxError: invalid syntax
```

**Solution:**
```bash
# Verify Python version
python --version

# If less than 3.8, update Python
# On Windows: download from python.org
# On Linux: sudo apt-get install python3.9
# On macOS: brew install python@3.9
```

### Problem 5: Matplotlib Backend Error

**Error:**
```
RuntimeError: Invalid DISPLAY variable
```

**Solution:**
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import plot2llm
```

### Problem 6: Memory Error

**Error:**
```
MemoryError
```

**Solution:**
```python
# Close figures after using them
import matplotlib.pyplot as plt
import plot2llm

fig, ax = plt.subplots()
# ... create figure ...
result = plot2llm.convert(fig)
plt.close(fig)  # Free memory
```

### Problem 7: SSL Error on Windows

**Error:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solution:**
```bash
# Update certificates
pip install --upgrade certifi

# Or use --trusted-host
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org plot2llm
```

---

## Advanced Configuration

### pip Configuration

Create `pip.conf` file (Linux/macOS) or `pip.ini` (Windows):

```ini
[global]
index-url = https://pypi.org/simple/
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
```

### Environment Configuration

Useful environment variables:

```bash
# Configure matplotlib backend
export MPLBACKEND=Agg

# Configure cache directory
export PYTHONPATH=/path/to/plot2llm:$PYTHONPATH

# Configure logging
export PLOT2LLM_LOG_LEVEL=INFO
```

### Jupyter Configuration

```python
# In a Jupyter cell
%matplotlib inline
import plot2llm

# Configure for better performance
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['figure.dpi'] = 100
```

### Docker Configuration

```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install plot2llm
RUN pip install plot2llm[all]

# Configure matplotlib
ENV MPLBACKEND=Agg

# Working directory
WORKDIR /app

# Default command
CMD ["python"]
```

### CI/CD Configuration

Example for GitHub Actions:

```yaml
name: Test Plot2LLM

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install plot2llm[all]
        pip install pytest
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v
```

---

## Uninstallation

### Uninstall plot2llm

```bash
pip uninstall plot2llm
```

### Uninstall with Dependencies

```bash
# Uninstall plot2llm and its dependencies
pip uninstall plot2llm numpy pandas matplotlib seaborn plotly -y
```

### Clean Cache

```bash
# Clean pip cache
pip cache purge

# Clean Python cache
python -Bc "import compileall; compileall.compile_dir('.', force=True)"
```

---

## Support

### Help Resources

- **Documentation**: [README.md](../README.md)
- **API Reference**: [API_REFERENCE.md](API_REFERENCE.md)
- **Examples**: [EXAMPLES.md](EXAMPLES.md)
- **Issues**: [GitHub Issues](https://github.com/your-username/plot2llm/issues)

### Diagnostic Commands

```bash
# System information
python -c "import sys; print(sys.version)"
python -c "import platform; print(platform.platform())"

# Plot2LLM information
python -c "import plot2llm; print(plot2llm.__version__)"

# Dependency information
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import pandas; print('pandas:', pandas.__version__)"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
```

### Reporting Issues

When reporting issues, include:

1. Python version
2. Plot2LLM version
3. Operating system
4. Code that reproduces the error
5. Complete error message
6. Stack trace if available

---

## Conclusion

This guide covers all aspects of Plot2LLM installation. If you encounter issues not covered here, consult the [Troubleshooting](#troubleshooting) section or open an issue on GitHub.

To start using Plot2LLM, consult the [Examples](EXAMPLES.md) and [API Reference](API_REFERENCE.md). 