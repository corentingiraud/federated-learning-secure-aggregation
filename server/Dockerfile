FROM python:3.6

WORKDIR /server

ADD requirements.txt .
RUN pip install -r requirements.txt

ADD . .

ENTRYPOINT ["python"]

CMD ["app/app.py"]
