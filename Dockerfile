FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && apt-get clean
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
