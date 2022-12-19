FROM pytorch/pytorch:1.7-cuda11.0-cudnn8-runtime

RUN pip install -r rquirements.txt

COPY . /app/

WORKDIR /app/src

CMD ["python", "/app/infer.py"]