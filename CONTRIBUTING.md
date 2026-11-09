# Contributing

Thanks for helping improve this project! Bug reports, new datasets, better models and docs are all welcome.

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Before you open a pull request
```bash
pytest tests                       # Python unit tests
node tests/test_parity.mjs         # browser engine == Python model
node tests/test_engine_robustness.mjs
```
If you change `improved/fnd.py` (text cleaning) you **must** mirror the change in `docs/engine.js`, then re-run
`python improved/export_web.py ...` so the parity test passes.

## Guidelines
- Keep results honest: always report accuracy on held-out *and* cross-dataset data, and keep source-leak stripping on.
- New datasets: remove overlap with existing data (see `improved/prepare_external.py`) and verify the label direction by reading samples.
- One topic per pull request; describe what you measured before and after.
- Be kind: see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
