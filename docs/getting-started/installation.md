# Installation

ScrollSwarm is lightweight and built in pure Python with minimal external dependencies. We recommend using [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management, though standard `pip` and virtual environments are fully supported.

---

## Prerequisites

- **Python:** Version `3.12` or newer
- **Operating System:** Linux, macOS, or Windows

---

## Option 1: Installation with `uv` (Recommended)

`uv` is an extremely fast Python package manager that handles virtual environment creation and dependency locking automatically.

### 1. Clone the Repository
```bash
git clone https://github.com/HamBa-m/scrollswarm.git
cd scrollswarm
```

### 2. Install Dependencies
```bash
uv sync
```
This creates a local `.venv` and installs all core dependencies (`gymnasium`, `pettingzoo`, `pygame`, `numpy`, and `pytest`).

### 3. Install Documentation Tools (Optional)
To build the documentation site locally with Material for MkDocs:
```bash
uv sync --extra docs
```

---

## Option 2: Installation with `pip`

If you prefer using standard Python `venv` and `pip`:

### 1. Clone & Create Environment
```bash
git clone https://github.com/HamBa-m/scrollswarm.git
cd scrollswarm
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 2. Install Package in Editable Mode
```bash
pip install -e .
```

To include documentation dependencies:
```bash
pip install -e ".[docs]"
```

---

## Verifying Your Installation

Run the automated test suite to verify that all modules, kinematics, sensors, and PettingZoo parallel compliance tests pass:

```bash
uv run pytest
# Or with standard venv activated:
pytest
```

Expected output:
```text
======================= 45 passed, 2 warnings in 1.30s =======================
```

---

## Next Steps

Now that ScrollSwarm is installed, head to the [Quickstart Guide](quickstart.md) to launch the interactive flight visualizer or run your first multi-agent stepping script.
