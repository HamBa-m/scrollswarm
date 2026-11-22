# Installation

ScrollSwarm is packaged as a standard Python package managed via `uv` or standard `pip`.

---

## 1. Using in a Separate Training Repository

To use ScrollSwarm as an external dependency in your downstream reinforcement learning project, install it directly from GitHub:

=== "Using uv (Recommended)"

    ```bash
    uv add git+https://github.com/HamBa-m/scrollswarm.git
    ```

=== "Using standard pip"

    ```bash
    pip install git+https://github.com/HamBa-m/scrollswarm.git
    ```

---

## 2. Local Editable Installation (Development)

To contribute to or modify ScrollSwarm locally, clone the repository and install it in editable mode:

```bash
# Clone the repository
git clone https://github.com/HamBa-m/scrollswarm.git
cd scrollswarm

# Install dependencies and sync virtual environment using uv
uv sync

# Run the test suite to verify installation
uv run pytest
```

---

## 3. Verifying Installation

Verify that the package and its vector assets load cleanly:

```bash
uv run python -c "import scrollswarm; print('ScrollSwarm version:', scrollswarm.__version__ if hasattr(scrollswarm, '__version__') else '0.1.0')"
```
