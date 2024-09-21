FROM public.ecr.aws/lambda/python:3.10
# Python docker images: https://github.com/docker-library/docs/tree/master/python/

COPY requirements.txt .
ENV HNSWLIB_NO_NATIVE=1

RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
COPY app.py ${LAMBDA_TASK_ROOT}

# EXPOSE 5001
CMD [ "app.lambda_handler" ]
