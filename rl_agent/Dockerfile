FROM python:3.11-slim

WORKDIR /app

# Install deps first (cache-friendly layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY *.py ./

# Pre-training checkpoint will be baked in at build time if present
# (CI runs pretrain.py before docker build, copies artifact here)
COPY checkpoints/ /checkpoints/

EXPOSE 8000

# Entrypoint: run pre-training if no checkpoint, then start server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
