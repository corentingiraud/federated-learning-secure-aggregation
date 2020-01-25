FROM python:3.6

WORKDIR /client

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD app .

ENTRYPOINT ["python"]

CMD ["client.py"]
