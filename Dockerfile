FROM anibali/pytorch:cuda-9.0

COPY --chown=user . /app/pt.darts/

WORKDIR /app/pt.darts/

RUN pip install pip -U && pip install -r requirements.txt
