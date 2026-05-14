# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a mathematical modeling project for Problem C of the Chinese National College Mathematical Contest in Modeling (CUMCM). The problem involves analysis and classification of ancient Chinese glass artifacts based on their chemical composition and physical properties.

## Data Files

@data/问题描述.md — Full problem statement (4 questions) with background
@data/数据说明.md — Data dictionary, field descriptions, key analysis points
@data/表单1_文物信息.csv — Artifact metadata (58 rows × 5 cols)
@data/表单2_化学成分.csv — Chemical compositions (69 sampling points × 15 cols)
@data/表单3_未知类别.csv — Unknown samples A1-A8 (8 rows × 16 cols)

## Problem-Solving Workflow

Problems are solved sequentially (1→4). For each problem:

1. **Discuss** — Analyze the problem with user, clarify approach, identify key decisions
2. **Phase-plan** — Break into steps, each with verifiable success criteria
3. **Implement** — Write code, verify each step before proceeding
4. **Document** — Sub-task analysis docs stored in `analysis/` directory

@analysis/problem1.md — Problem 1 analysis and sub-task steps (when created)
@analysis/problem2.md — Problem 2 analysis and sub-task steps (when created)
@analysis/problem3.md — Problem 3 analysis and sub-task steps (when created)
@analysis/problem4.md — Problem 4 analysis and sub-task steps (when created)

## Development Environment

- **Python 3.11** managed by **uv** (v0.9+)
- **Package manager**: uv (fast venv + dependency resolution)
- **Lint/Format**: ruff, **Type check**: pyright

### Quickstart

```powershell
uv sync                    # Install all dependencies
uv run python script.py    # Run a script
uv run jupyter lab         # Start JupyterLab
uv run ruff check .        # Lint
uv run pyright .           # Type check
uv add <package>           # Add a dependency
uv add --dev <package>     # Add a dev dependency
```

### Key Libraries

- pandas, numpy — data processing
- matplotlib — visualization
- scipy, scikit-learn — statistical analysis, classification
- openpyxl — read .xlsx files
- jupyter — interactive notebooks
- rich — terminal output formatting
