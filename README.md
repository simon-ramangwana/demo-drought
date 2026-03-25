# RozviDrought Demo Inference System

## Overview

This repository demonstrates how to run drought inference using the **rozvidrought ecosystem**.

The system converts spatial inputs into drought predictions using packaged models.

Pipeline:

Geometry  
→ Data extraction  
→ Feature engineering  
→ Subsystem models  
→ Fusion model  
→ Drought classification

This repository is a **reference implementation** showing how to connect:

- rozvidrought-datasets
- rozvidrought-inputs
- rozvidrought-subsystems
- rozvidrought

into a working API and frontend.

---

## Important Rule

**Do not start from the repository root.**

Always start inside:


RozviDrought/


All runtime code lives there.

---

## Repository Structure


RozviDrought/
│
├── app/
│ ├── api/
│ ├── services/
│ ├── schemas/
│ └── frontend/
│
├── provenance/
├── runs/
├── tests/
├── docs/
│
├── run_api.py
├── requirements.txt
└── README.md


---

## System Requirements

Python:


Python 3.10 or newer


Operating system:


Windows, Linux, or macOS


---

## Installation

Clone the repository.


git clone https://github.com/simon-ramangwana/demo-drought.git


Move into the application folder.


cd demo-drought
cd RozviDrought


Install dependencies.


pip install -r requirements.txt


---

## Required Data

Large datasets are **not stored in Git**.

You must obtain them separately.

---

## Required Dataset

The system requires one serving dataset:


master_inputs_long_198001_205012.parquet


---

## Where to Place the Parquet File

Place the dataset here:


RozviDrought/data/master_inputs/


Example:


RozviDrought/
data/
master_inputs/
master_inputs_long_198001_205012.parquet


This file is the **serving dataset** used during inference.

The API reads this file directly.

---

## What the Dataset Contains

The dataset contains:


pixel_id
row
col
lon
lat
scenario
yyyymm
t2m
d2m
pet
sm
ndvi
tws


Each row represents:


one pixel
one month


---

## Start the API

From inside:


RozviDrought/


Run:


python run_api.py


You should see:


Server running at http://localhost:8000


---

## Open the Demo Interface

Open your browser.


http://localhost:8000


You can:

- click a location
- select scenario
- select month
- view drought classification

---

## API Endpoint


GET /infer/point


Example:


http://localhost:8000/infer/point?lon=30.344238&lat=-18.788573&scenario=ssp245&yyyymm=202701


---

## Supported Geometry Types

The packages support:

- Point
- Polygon
- MultiPolygon

The demo interface currently demonstrates:


Point inference


Additional geometry examples will be provided in:


RozviDrought/docs/


---

## Running Custom Scripts

Developers can create their own scripts anywhere inside:


RozviDrought/


Example:


RozviDrought/scripts/
RozviDrought/tests/
RozviDrought/docs/
RozviDrought/custom/


The system does not restrict script locations.

---

## How Inference Works Internally

The runtime flow:


coordinates
→ grid locator
→ pixel lookup
→ timeseries retrieval
→ feature creation
→ subsystem predictions
→ fusion prediction
→ drought result


---

## Models Used

Subsystem models:


Atmospheric
Soil Moisture
Vegetation
Hydrology


Fusion models:


Logistic stacking
Hybrid fusion


---

## Output Example


{
"drought_class": "moderate",
"probabilities": {
"normal": 0.21,
"moderate": 0.58,
"severe": 0.21
},
"confidence": 0.58
}


---

## Logs

Logs are written to:


RozviDrought/runs/logs/


---

## Provenance

All model and dataset lineage information is stored in:


RozviDrought/provenance/


This includes:

- manifests
- schemas
- validation records
- processing notes

---

## Testing

Run tests:


pytest


Or run smoke test:


python tests/test_smoke.py


---

## Common Issues

### Dataset Not Found

Check:


RozviDrought/data/master_inputs/


---

### Server Does Not Start

Check:


pip install -r requirements.txt


---

### No Inference Output

Check:

- dataset exists
- correct scenario
- correct month

---

## Development Workflow

Typical developer workflow:


git pull
pip install -r requirements.txt
cd RozviDrought
python run_api.py


---

## Next Documentation

Additional examples will be provided in:


RozviDrought/docs/


Planned documents:


point_inference_script.py
polygon_inference_script.py
multipolygon_inference_script.py
batch_inference_script.py


---

## System Summary

The models already exist.

The datasets already exist.

The extraction pipeline already exists.

This repository shows:

- how to run inference
- how to connect models
- how to build applications
- how to visualize drought outputs
