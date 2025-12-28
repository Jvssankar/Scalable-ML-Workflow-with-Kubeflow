FROM python:3.9-slim

RUN pip install tensorflow==2.20.0 numpy

COPY train.py /train.py

CMD ["python", "/train.py"]