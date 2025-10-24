FROM python:3.12-slim

# faster, smaller installs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# install system deps only if you later need them (kept minimal here)
# RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# copy deps first to leverage Docker layer cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the repo (src, models, bin, docs, etc.)
COPY . .

# make sure Python can import from /app
ENV PYTHONPATH=/app

# simple healthcheck using Python (no curl needed)
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000
# use API_KEY to protect the endpoints (optional but recommended)
# set API_KEY at runtime: docker run -e API_KEY=change-me ...
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
