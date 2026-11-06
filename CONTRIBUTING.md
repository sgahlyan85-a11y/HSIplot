# Contributing

Bug reports and pull requests are welcome.

- Open an issue with the command you ran, the full error text and the output of `hsiplot --version` and `python --version`.
- For SDK problems include the output of `hsiplot check <file>.lo`.
- Development install: `pip install -e ".[dev]"`; run `ruff check .` and `pytest` before a pull request.
- A change to a statistical method needs a test with a known answer (a positive and a negative control) and a reference in `docs/METHODS.md`.
- Do not commit data (`.lo`, `.npz`, spectra tables) or any Living Optics SDK files.
