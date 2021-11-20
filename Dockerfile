FROM huggingface/transformers-pytorch-cpu:latest

COPY ./ /app
WORKDIR /app

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG S3_STORAGE_URL

# aws credentials configuration
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY\
    S3_STORAGE_URL=$REGISTRY_URL

# install requirements
RUN pip install "dvc[s3]"   # Since s3 is the remote storage
RUN pip install -r requirements.txt

# initialise dvc
RUN dvc init --no-scm

# configuring remote server in dvc
RUN dvc remote add -d s3-store $S3_STORAGE_URL

# pulling the trained model
RUN dvc pull dvcfiles/model.dvc

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# running the application
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]