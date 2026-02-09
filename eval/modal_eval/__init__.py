"""
Modal-based synthesis evaluation module.

This module provides parallel synthesis evaluation using Modal's cloud infrastructure.

Usage:
    # Run synthesis test
    modal run modal_eval/synthesis_app.py

    # Use with run_full_eval.py
    python eval/run_full_eval.py frozen_lake --train-mode var_config --use-modal
"""

from pathlib import Path

MODULE_DIR = Path(__file__).parent
PROJECT_ROOT = MODULE_DIR.parent
