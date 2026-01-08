---
title: "Contributing"
description: "How to contribute to halo forge"
---

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch
4. Make your changes
5. Submit a pull request

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/halo-forge.git
cd halo-forge/toolbox
./build.sh

toolbox enter halo-forge
pip install -e ".[dev]"
```

## Running Tests

```bash
# Quick smoke test
halo-forge test --level smoke

# Full test suite
pytest tests/
```

## Code Style

We use:
- `black` for formatting
- `isort` for imports
- Type hints where practical

```bash
black halo_forge/
isort halo_forge/
```

## Pull Request Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Update documentation as needed
- Follow existing code style

## Areas for Contribution

- **New verifiers**: Rust, Go, TypeScript, etc.
- **Dataset support**: Additional public datasets
- **Performance**: Optimization opportunities
- **Documentation**: Examples, tutorials
- **Bug fixes**: Check GitHub issues

## Reporting Issues

Include:
- halo-forge version
- Hardware info (`halo-forge info`)
- Steps to reproduce
- Error messages
- Config files (sanitized)

## License

By contributing, you agree that your contributions will be licensed under Apache 2.0.
