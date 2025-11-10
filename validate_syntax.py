#!/usr/bin/env python3
"""
Syntax validation script to check if all new BAMT 2.0.0 modules have valid Python syntax.
This script doesn't require dependencies to be installed.
"""

import py_compile
import os
from pathlib import Path


def find_python_files(base_dir):
    """Find all Python files in the specified directory."""
    python_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files


def validate_syntax(file_path):
    """Validate the syntax of a Python file."""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)


def main():
    """Run syntax validation on all new BAMT 2.0.0 modules."""
    print("=" * 70)
    print("BAMT 2.0.0 Syntax Validation")
    print("=" * 70)
    
    # Directories to check
    base_dirs = [
        'bamt/core',
        'bamt/dag_optimizers',
        'bamt/score_functions',
        'bamt/parameter_estimators',
        'bamt/models'
    ]
    
    all_valid = True
    total_files = 0
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"\n⚠ Directory not found: {base_dir}")
            continue
        
        print(f"\n📁 Checking {base_dir}...")
        python_files = find_python_files(base_dir)
        
        for file_path in python_files:
            total_files += 1
            is_valid, error = validate_syntax(file_path)
            
            if is_valid:
                print(f"  ✓ {os.path.relpath(file_path)}")
            else:
                print(f"  ✗ {os.path.relpath(file_path)}")
                print(f"    Error: {error}")
                all_valid = False
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Total files checked: {total_files}")
    
    if all_valid:
        print("\n✓ All files have valid Python syntax!")
        return 0
    else:
        print("\n✗ Some files have syntax errors. Please check the errors above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
