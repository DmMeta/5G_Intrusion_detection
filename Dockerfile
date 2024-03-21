FROM python:latest

RUN pip3 install --upgrade pip


RUN mkdir -p /opt/model_server/src/
RUN mkdir -p /opt/model_server/models/
WORKDIR /opt/model_server
COPY models/ models/
WORKDIR /opt/model_server/src/
COPY src/ .
COPY ./src/server_requirements.txt .
RUN pip3 install -r server_requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

EXPOSE 8891


ENTRYPOINT ["uvicorn","server:app","--reload","--host","0.0.0.0","--port","8891"]