FROM python:latest

RUN pip3 install --upgrade pip
RUN pip3 install uvicorn fastapi[all] 

RUN mkdir -p /opt/model_server/src/
RUN mkdir -p /opt/model_server/models/
WORKDIR /opt/model_server
COPY models/ models/
WORKDIR /opt/model_server/src/

EXPOSE 8891


ENTRYPOINT ["uvicorn","server:app","--reload","--host","0.0.0.0","--port","8891"]