FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt \
    && python -m spacy download en_core_web_sm

COPY . /app/api/
ENV PYTHONPATH=/app
EXPOSE 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
