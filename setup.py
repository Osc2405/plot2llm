from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="plot2llm",
    version="0.1.20",  # Beta version
    author="Osc2405",
    author_email="orosero2405@gmail.com",
    description="Convert Python figures to LLM-readable formats. Supports matplotlib and seaborn with multiple output formats.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Osc2405/plot2llm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.0.0",
        "seaborn>=0.11.0",
        "numpy>=1.9.0",
        "pandas>=1.1.0",
        "webcolors>=1.11.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
            "isort>=5.10.0",
            "pre-commit>=2.15.0",
        ],
    },
)
