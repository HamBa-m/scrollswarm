# Contributing to ScrollSwarm

Thank you for your interest in improving ScrollSwarm! Contributions from the research, robotics, and reinforcement learning communities are welcome.

## Development Workflow

ScrollSwarm is managed using [uv](https://github.com/astral-sh/uv) for fast, reproducible virtual environment management.

### 1. Environment Setup

```bash
git clone https://github.com/HamBa-m/scrollswarm.git
cd scrollswarm
uv sync
```

### 2. Running Verification & Unit Tests

Before submitting changes, ensure the entire test suite passes:

```bash
uv run pytest
```

### 3. Building Documentation

Verify that documentation builds strictly without errors:

```bash
uv run mkdocs build --strict
```

To preview documentation locally:

```bash
uv run mkdocs serve
```

### 4. Code Standards

- **Formatting & Typing:** Write clean, modular Python with standard PEP 8 conventions and clear type annotations.
- **White-Box Clarity:** Avoid hidden assumptions or undocumented dependencies.
- **Test Coverage:** Accompany new features, environment mechanics, or mathematical models with corresponding unit tests in `tests/`.

### 5. Pull Requests

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Commit your changes: `git commit -m "Add descriptive commit message"`
3. Push to your fork and submit a Pull Request to `main`.
