"""For relative imports to work in Python >= 3.6"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
