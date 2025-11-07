# GPU-enabled TF base image
FROM tensorflow/tensorflow:2.20.0-gpu

# System
ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY src /app/src
COPY configs /app/configs

# (Optional) copy data if you want a tiny smoke test dataset baked in
# COPY data /app/data

# Default command is unused in Flyte (tasks run your functions)
CMD ["python", "-c", "print('Container ready')"]
