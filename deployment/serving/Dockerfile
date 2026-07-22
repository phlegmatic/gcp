# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Cloud Run inference image for the demand-forecasting ensemble.
#
# Design notes (free-tier / cost-aware):
#  * python:3.11-slim base -> small image, fast cold start on Cloud Run.
#  * Runtime deps are PINNED to the SAME versions used by Pipeline 2's training
#    component (scikit-learn 1.5.1 / pandas 2.2.2 / numpy 1.26.4) so the pickled
#    estimators unpickle without version-skew warnings or errors. This is a real
#    production footgun and the pins are deliberate.
#  * Non-root user; app binds to $PORT (Cloud Run injects it, default 8080).
#  * Only the `src/` package is copied -> no dbt/notebooks/tests in the image.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install runtime dependencies. Versions MUST match the training component pins.
RUN pip install --no-cache-dir \
    "scikit-learn==1.5.1" \
    "pandas==2.2.2" \
    "numpy==1.26.4" \
    "pyarrow==16.1.0" \
    "google-cloud-storage==2.17.0" \
    "python-dotenv==1.0.1" \
    "fastapi==0.111.1" \
    "uvicorn[standard]==0.30.3"

# Copy only the installable package source (keeps the image lean).
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package itself WITHOUT re-resolving heavy deps (already pinned
# above). --no-deps avoids pulling KFP / aiplatform, which serving never needs.
RUN pip install --no-cache-dir --no-deps .

# Drop privileges.
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Cloud Run sends traffic to $PORT. Single worker + high async concurrency is
# ideal for cheap CPU-bound sklearn predicts; scale-out is handled by Cloud Run.
CMD ["sh", "-c", "uvicorn forecasting.serving.app:app --host 0.0.0.0 --port ${PORT}"]
