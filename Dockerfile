FROM python:3.11-slim
WORKDIR /workspace/backend
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .
RUN mkdir -p /workspace/data /workspace/models
ENV PYTHONPATH=/workspace/backend
EXPOSE 8000
