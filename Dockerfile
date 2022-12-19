# Training
FROM python:3.9

WORKDIR /app

# Copy necessary files
COPY data/ /app/data/
COPY myproject/ /app/myproject/
COPY Makefile /app
COPY poetry.lock /app
COPY pyproject.toml /app
COPY README.md /app

# Install deps
RUN make init

# install packages
RUN poetry install --no-root
#VOLUME [ "/app" ]

# Train model
CMD poetry run python myproject/run.py
