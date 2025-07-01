#!/usr/bin/env python3
"""
Script to organize all example files, test outputs, and validation results
into a structured directory outside the main repository.
"""

import os
import shutil
import glob
from pathlib import Path

def create_directory_structure():
    """Create the organized directory structure."""
    base_dir = Path("plot2llm_examples")
    
    # Create main directories
    directories = [
        base_dir / "examples" / "seaborn",
        base_dir / "examples" / "advanced",
        base_dir / "examples" / "validation",
        base_dir / "tests" / "outputs",
        base_dir / "tests" / "debug",
        base_dir / "validation" / "results",
        base_dir / "validation" / "images",
        base_dir / "docs" / "screenshots"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return base_dir

def move_files(base_dir):
    """Move files to their appropriate directories."""
    
    # Move seaborn examples
    seaborn_examples = [
        "examples_seaborn/*.png",
        "examples_seaborn/*.json"
    ]
    
    for pattern in seaborn_examples:
        for file_path in glob.glob(pattern):
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                dest_path = base_dir / "examples" / "seaborn" / filename
                shutil.move(file_path, dest_path)
                print(f"📁 Moved: {file_path} → {dest_path}")
    
    # Move test outputs
    test_patterns = [
        "test_*.json",
        "test_*.png",
        "debug_*.json",
        "debug_*.png"
    ]
    
    for pattern in test_patterns:
        for file_path in glob.glob(pattern):
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                if file_path.startswith("debug_"):
                    dest_path = base_dir / "tests" / "debug" / filename
                else:
                    dest_path = base_dir / "tests" / "outputs" / filename
                shutil.move(file_path, dest_path)
                print(f"📁 Moved: {file_path} → {dest_path}")
    
    # Move example files
    example_patterns = [
        "example_*.json",
        "example_*.png"
    ]
    
    for pattern in example_patterns:
        for file_path in glob.glob(pattern):
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                dest_path = base_dir / "examples" / "validation" / filename
                shutil.move(file_path, dest_path)
                print(f"📁 Moved: {file_path} → {dest_path}")
    
    # Move advanced examples
    advanced_patterns = [
        "examples_advanced/*.json",
        "examples_advanced/*.png"
    ]
    
    for pattern in advanced_patterns:
        for file_path in glob.glob(pattern):
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                dest_path = base_dir / "examples" / "advanced" / filename
                shutil.move(file_path, dest_path)
                print(f"📁 Moved: {file_path} → {dest_path}")

def create_readme(base_dir):
    """Create a README file for the examples directory."""
    readme_content = """# Plot2LLM Examples and Validation Results

This directory contains all examples, test outputs, and validation results for the Plot2LLM library.

## Directory Structure

### 📁 examples/
- **seaborn/**: Seaborn-specific examples and outputs
- **advanced/**: Advanced usage examples
- **validation/**: Validation test results

### 📁 tests/
- **outputs/**: Test output files (JSON and images)
- **debug/**: Debug files and intermediate results

### 📁 validation/
- **results/**: Validation result files
- **images/**: Generated validation images

### 📁 docs/
- **screenshots/**: Documentation screenshots

## File Types

- **.json**: Analysis results in JSON format
- **.png**: Generated plot images
- **.txt**: Log files and text outputs

## Usage

These files are generated automatically when running:
- `python example_seaborn.py`
- `python test_improvements.py`
- `python debug_heatmap_detection.py`
- Other example and test scripts

## Note

This directory is excluded from the main repository via .gitignore to keep the repository clean.
"""
    
    readme_path = base_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"📝 Created README: {readme_path}")

def cleanup_empty_directories():
    """Remove empty directories that were created by examples."""
    empty_dirs = [
        "examples_seaborn",
        "examples_advanced",
        "examples_validation"
    ]
    
    for dir_name in empty_dirs:
        if os.path.exists(dir_name) and not os.listdir(dir_name):
            os.rmdir(dir_name)
            print(f"🗑️  Removed empty directory: {dir_name}")

def main():
    """Main function to organize all files."""
    print("🗂️  Organizing Plot2LLM examples and outputs...")
    print("=" * 50)
    
    # Create directory structure
    base_dir = create_directory_structure()
    
    # Move files
    move_files(base_dir)
    
    # Create README
    create_readme(base_dir)
    
    # Cleanup empty directories
    cleanup_empty_directories()
    
    print("\n" + "=" * 50)
    print("✅ Organization complete!")
    print(f"📁 All files organized in: {base_dir}")
    print("\n📋 Summary:")
    print("  - Examples moved to plot2llm_examples/examples/")
    print("  - Test outputs moved to plot2llm_examples/tests/")
    print("  - Validation results moved to plot2llm_examples/validation/")
    print("  - Empty directories cleaned up")
    print("  - README created with documentation")

if __name__ == "__main__":
    main() 