FROM python

WORKDIR /app

COPY /FMClassifier .

RUN pip3 install -r requirements.txt