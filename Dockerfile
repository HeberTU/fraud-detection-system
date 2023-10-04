FROM python:3.8.5

RUN pip3 install poetry
RUN poetry config virtualenvs.create false

RUN mkdir -p /cronos
WORKDIR /cronos

COPY pyproject.toml .
RUN poetry install

COPY . .

CMD ["python", "train.py"]