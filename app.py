"""
ACDP Tree — Streamlit Dashboard Entry Point.

Usage:
    streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frontend.dashboard import main

if __name__ == "__main__":
    main()
