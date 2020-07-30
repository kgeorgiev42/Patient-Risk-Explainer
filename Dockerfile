FROM python:3.7

# This prevents Python from writing out pyc files
# ENV PYTHONDONTWRITEBYTECODE 1
# This keeps Python from buffering stdin/stdout
# ENV PYTHONUNBUFFERED 1

MAINTAINER Konstantin S. Georgiev "dragonflareful@gmail.com"

# install system dependencies
RUN apt-get update \
    && apt-get -y install gcc make \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --no-cache-dir --upgrade pip

COPY . /patient-risk-model-explainer-ui

WORKDIR /patient-risk-model-explainer-ui

# install project requirements
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# ENTRYPOINT [ "python" ]

# Run app.py when the container launches
# CMD [ "application.py","run","--host","0.0.0.0" ]
CMD [ "python", "application.py" ]