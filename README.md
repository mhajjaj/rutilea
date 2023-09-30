# Description
This repository is used to run the YOLONAS project for Gear Inspection Dataset.

# Installation
Please install all the required packages, like:
1. super_gradients
2. torch
3. ..

###### Note: maybe you need to fix the compatible packages (to bu updated).`

# How to run
- Got to root and run `aisin_gear_inspection\src\GearInspection.py`
- Please create this structure of the Dataset:
#### GearInspection-Dataset Directory Structure
> GearInspection-Dataset \
├── GategoryNG \
│ ├── Class1 \
│ │ ├── test \
│ │ │ ├── images/ \
│ │ │ └── labels/ \
│ │ ├── train \
│ │ │ ├── images/ \
│ │ │ └── labels/ \
│ │ ├── valid \
│ │ │ ├── images/ \
│ │ │ └── labels/ \
│ │ ├── data.yaml \
│ ├── Class2 \
│ │ ├── test \
│ │ │ ├── images/ \
│ │ │ └── labels/ \
│ │ ├── train \
│ │ │ ├── images/ \
│ │ │ └── labels/ \
│ │ ├── valid \
│ │ │ ├── images/ \
│ │ │ └── labels/ \
│ │ ├── data.yaml \
│ ├── Class3 \
│ │ ├── test \
│ │ │ ├── images/ \
│ │ │ └── labels/ \
│ │ ├── train \
│ │ │ ├── images/ \
│ │ │ └── labels/ \
│ │ ├── valid \
│ │ │ ├── images/ \
│ │ │ └── labels/ \
│ │ └── data.yaml \
│ ├── ClassAll \
│ │ ├── test \
│ │ │ ├── images/ \
│ │ │ └── labels/ \
│ │ ├── train \
│ │ │ ├── images/ \
│ │ │ └── labels/ \
│ │ ├── valid \
│ │ │ ├── images/ \
│ │ │ └── labels/ \
│ │ └── data.yaml \
└── README.md 

- The folder has the `data.yaml`
- Please use GearInspectionUserInterface.py to run the generated model for client.