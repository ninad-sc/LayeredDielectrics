"""Run from a source checkout without installation.

Edit ANGLES_DEG in src/layered_dielectrics/plot_skin_reflection.py.
"""
from pathlib import Path
import sys

# Resolve relative to this file so launchers in IDEs also find the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from layered_dielectrics.plot_skin_reflection import main

if __name__ == '__main__':
    main()
