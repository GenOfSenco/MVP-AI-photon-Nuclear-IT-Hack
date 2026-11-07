FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.5.1+cpu' \
 && python -m pip install --no-cache-dir -r requirements.txt
COPY submission/ submission/
COPY weights/ weights/
COPY examples/ examples/
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin doe
USER doe
RUN python -c "from submission.backend import Engine; Engine()"
EXPOSE 7860
CMD ["python", "-m", "submission.web"]
