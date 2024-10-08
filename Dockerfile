FROM python:3.8
# Python docker images: https://github.com/docker-library/docs/tree/master/python/

USER root

# Copy the src
WORKDIR /app
COPY src/ /app/src/
COPY ./requirements.txt /app
COPY ./.env /app
RUN ls -la /app

RUN mkdir -p /chromadb
RUN mkdir -p /.cache
RUN mkdir -p /.phoenix
# Install python dependencies
RUN chown 1001 /chromadb
RUN chown 1001 /.phoenix
RUN chown 1001 /.cache
RUN python3 --version
RUN pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt

USER 1001

# EXPOSE 5001
ENTRYPOINT ["python3", "/app/src/app.py"]
