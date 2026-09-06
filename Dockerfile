# Freeze the exact Python version inside the image so the host (Render, AWS, local)
# can never pick a different one. This permanently prevents the "no wheel -> build
# from source -> metadata-generation-failed" class of errors.
FROM python:3.12.7-slim

# Keep Python output unbuffered and skip .pyc files for cleaner container logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first (better layer caching). We force prebuilt wheels for
# the packages that contain compiled C/C++ code, so pip fails loudly if a wheel is
# missing rather than silently compiling from source (the root cause of past
# metadata-generation-failed errors). Pure-Python packages install normally.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install \
      --only-binary=numpy,scipy,pandas,scikit-learn,xgboost,matplotlib,pillow \
      -r requirements.txt

# Copy the rest of the project (respects .dockerignore).
COPY . .

# Render provides the port via the $PORT env var. Default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

# Use shell form so $PORT expands at runtime.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
