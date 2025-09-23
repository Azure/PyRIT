# PyRIT - SQLAlchemy 1.4 Compatible Version

This is a modified version of [PyRIT (Python Risk Identification Tool for LLMs)](https://github.com/Azure/PyRIT) that has been patched to work with SQLAlchemy 1.4.47 instead of SQLAlchemy 2.0+.

## Overview

PyRIT is a library used to assess the robustness of Large Language Models (LLMs). This fork maintains all the functionality of the original PyRIT but makes it compatible with environments that require SQLAlchemy 1.4.x.

## Key Modifications

- Modified database models to work with SQLAlchemy 1.4.47
- Added custom UUID type implementation for SQLAlchemy 1.4
- Replaced SQLAlchemy 2.0-specific features with 1.4-compatible alternatives
- Added compatibility with pydantic-sqlalchemy 0.0.9

## Installation

```bash
pip install pyrit-sqlalchemy14
```

Or with Poetry:

```bash
poetry add pyrit-sqlalchemy14
```

## Requirements

- Python 3.10 or higher
- SQLAlchemy 1.4.47 (not compatible with SQLAlchemy 2.0+)
- pydantic-sqlalchemy 0.0.9

## Usage

The usage is identical to the original PyRIT library. Please refer to the [official PyRIT documentation](https://github.com/Azure/PyRIT) for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original PyRIT developed by Microsoft AI Red Team
- This SQLAlchemy 1.4 compatible version is maintained independently
