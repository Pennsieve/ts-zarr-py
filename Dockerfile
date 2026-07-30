FROM python:3.12

WORKDIR /app

RUN apt clean && apt-get update && apt-get -y install libhdf5-dev

COPY ts_zarr/requirements.txt /app/ts_zarr/requirements.txt

RUN pip install -r /app/ts_zarr/requirements.txt

COPY ts_zarr/ /app/ts_zarr

ENV PYTHONPATH="/app"

CMD ["python3.12", "-m", "ts_zarr.main"]
