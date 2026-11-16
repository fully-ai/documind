FROM python:3.12-slim

WORKDIR /app
RUN useradd --create-home --shell /bin/bash appuser
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

USER appuser
ENV DOCUMIND_HOST=0.0.0.0 \
    DOCUMIND_PORT=8092 \
    DOCUMIND_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8092
CMD ["documind"]
