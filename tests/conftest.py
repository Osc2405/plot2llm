"""
Pytest configuration and shared fixtures for plot2llm tests.
"""

import pytest
import matplotlib.pyplot as plt
import matplotlib
import warnings
import numpy as np


def pytest_configure(config):
    """Configure pytest environment."""
    # Use non-interactive backend for matplotlib
    matplotlib.use("Agg")

    # Suppress warnings during tests
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

    # Turn off interactive mode
    plt.ioff()


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Automatically clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    np.random.seed(42)  # For reproducible tests
    return {
        "x_simple": [1, 2, 3, 4, 5],
        "y_simple": [2, 4, 6, 8, 10],
        "x_scatter": np.random.rand(50),
        "y_scatter": np.random.rand(50),
        "categories": ["A", "B", "C", "D"],
        "values": [10, 25, 15, 30],
        "normal_data": np.random.normal(0, 1, 1000),
        "multivariate": {
            "x": np.random.rand(100),
            "y": np.random.rand(100),
            "z": np.random.rand(100),
        },
    }


@pytest.fixture
def basic_figure():
    """Create a basic matplotlib figure for testing."""
    fig, ax = plt.subplots(figsize=(8, 6))
    return fig, ax


@pytest.fixture
def subplot_figure():
    """Create a figure with multiple subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    return fig, axes


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "edge_case: mark test as edge case test")
