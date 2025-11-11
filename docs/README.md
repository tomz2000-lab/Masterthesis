# Documentation

This directory contains the Sphinx documentation for the Negotiation Platform.

## Building Documentation Locally

### Prerequisites

Install the documentation requirements:

```bash
pip install -r docs/requirements.txt
```

### Build HTML Documentation

**On Linux/Mac:**
```bash
make html
```

**On Windows:**
```batch
make.bat html
```

The built documentation will be available in `docs/_build/html/index.html`.

### Other Output Formats

- **PDF**: `make latexpdf`
- **ePub**: `make epub`
- **Clean build**: `make clean`

## Read the Docs Integration

This documentation is configured to build automatically on Read the Docs when pushed to the main branch.

### Configuration Files

- `.readthedocs.yaml` - Read the Docs build configuration
- `docs/conf.py` - Sphinx configuration
- `docs/requirements.txt` - Documentation build dependencies

## Documentation Structure

- `index.rst` - Main documentation page
- `installation.rst` - Installation instructions
- `quickstart.rst` - Quick start guide
- `configuration.rst` - Configuration documentation
- `games.rst` - Game documentation
- `examples.rst` - Usage examples
- `api/` - Auto-generated API documentation

## Contributing to Documentation

1. Edit the `.rst` files in the `docs/` directory
2. For API documentation, ensure docstrings follow Google style
3. Build locally to test changes: `make html`
4. Commit and push - Read the Docs will build automatically

## Docstring Style

Use Google-style docstrings as shown in your `compare_games_statistics_FIXED.py` file:

```python
def your_function(param1, param2):
    """
    Brief description of the function.

    Args:
        param1 (type): Description of param1.
        param2 (type): Description of param2.

    Returns:
        type: Description of return value.

    Raises:
        ExceptionType: Description of when this exception is raised.

    Example:
        >>> result = your_function("test", 42)
        >>> print(result)
        expected_output
    """
```