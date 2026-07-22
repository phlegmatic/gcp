"""Online model serving: pure inference core + FastAPI app.

Layering rule (see PROJECT_OVERVIEW.md §5):
- `predictor.py` is PURE (no web/cloud framework) so it is unit-testable and
  reuses `forecasting.models.features` to guarantee train/serve parity.
- `app.py` is the thin web layer (FastAPI) that loads the artifact and delegates
  all real work to `predictor.py`.
"""
