# TerraSignal: Crop Stress Platform

ML platform for predicting crop stress in UK arable fields using Sentinel-2 satellite imagery, weather, and soil data.

## Architecture

![Architecture](docs/architecture.png)

## Overview

```
Sentinel-2 bands + weather + soil → XGBoost model → stress_index (0–1)
```

Stress levels: **low** < 0.3 · **moderate** 0.3–0.6 · **high** > 0.6

## Project Structure

```
crop-stress-platform/
├── src/
│   ├── ingestion/
│   │   ├── sentinel_stac.py      # Sentinel-2 via CDSE STAC + S3
│   │   ├── weather.py            # Open-Meteo daily weather
│   │   └── soilgrids.py          # SoilGrids REST API
│   ├── processing/
│   │   ├── cloud_mask.py         # SCL-based cloud masking
│   │   ├── raster_clip.py        # AOI clipping
│   │   ├── spectral_indices.py   # NDVI, EVI, NDWI, NDRE, SAVI
│   │   └── zonal_statistics.py   # Per-field aggregation
│   ├── features/
│   │   ├── satellite_features.py # Satellite feature builder
│   │   ├── weather_features.py   # Weather feature builder
│   │   └── build_features.py     # Combined feature matrix
│   ├── training/
│   │   ├── train.py              # XGBoost training
│   │   ├── evaluate.py           # Metrics
│   ├── inference/
│   │   ├── handler.py            # AWS Lambda entry point
│   │   └── predictor.py          # Model loading + prediction
│   ├── extract.py                # Unified data extraction
│   └── build_dataset.py          # Dataset builder for training
├── dashboard/
│   └── app.py                    # Streamlit UI (calls Lambda via HTTP)
├── pipelines/
│   └── crop_stress_pipeline.py   # End-to-end local pipeline
├── infrastructure/
│   ├── template.yaml             # AWS SAM template
│   └── state_machine.json        # Step Functions definition
├── data/
│   ├── locations.csv             # Field locations
│   └── locations_uk_fields.csv   # UK field locations
├── tests/
│   ├── test_geometry.py
│   ├── test_indices.py
│   └── test_no_leakage.py
├── Dockerfile
├── requirements.txt
├── requirements-dashboard.txt
└── conftest.py
```

## Setup

```bash
conda create -n agriai python=3.11
conda activate agriai
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
CDSE_S3_ACCESS_KEY=your-cdse-access-key
CDSE_S3_SECRET_KEY=your-cdse-secret-key
MODEL_PATH=s3://your-bucket/models/crop_stress_model.json
APP_PASSWORD=your-dashboard-password
```

## Docker

### Build

```bash
docker build -t crop-stress:latest .
```

### Run locally

```bash
docker run -p 9000:8080 \
  -e CDSE_S3_ACCESS_KEY=your-key \
  -e CDSE_S3_SECRET_KEY=your-secret \
  -e MODEL_PATH=s3://your-bucket/models/crop_stress_model.json \
  crop-stress:latest
```

Test the local container:

```bash
curl -X POST http://localhost:9000/2015-03-31/functions/function/invocations \
  -H "Content-Type: application/json" \
  -d '{"lat": 51.5, "lon": -1.2, "start_date": "2024-04-01", "end_date": "2024-06-30"}'
```

### Push to ECR

```bash
aws ecr get-login-password --region eu-west-1 | \
  docker login --username AWS --password-stdin 582367504589.dkr.ecr.eu-west-1.amazonaws.com

docker tag crop-stress:latest 582367504589.dkr.ecr.eu-west-1.amazonaws.com/crop-stress:latest
docker push 582367504589.dkr.ecr.eu-west-1.amazonaws.com/crop-stress:latest
```

> CI/CD: pushing to the `development` branch triggers GitHub Actions to build and push automatically.
