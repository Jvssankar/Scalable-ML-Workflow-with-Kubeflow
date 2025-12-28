FROM python:3.9-slim

RUN pip install tensorflow flask numpy

COPY model.py /model.py

CMD ["python", "/model.py"]
