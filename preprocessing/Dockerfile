FROM python:3.9-slim

RUN pip install tensorflow numpy

COPY preprocess.py /preprocess.py

ENTRYPOINT ["python", "/preprocess.py"]
