FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime
WORKDIR /worker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY worker.py .
ENV HF_HOME=/models
EXPOSE 8010
CMD ["uvicorn","worker:app","--host","0.0.0.0","--port","8010","--no-access-log"]
