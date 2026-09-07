# Viewer only. outputs/ and sample/ come in as bind mounts (see docker-compose.yaml),
# so they are deliberately not COPYed -- outputs/ alone is ~160 MB.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config.yaml .

EXPOSE 8765
CMD ["python", "-m", "src.viewer", "--host", "0.0.0.0", "--no-open"]
