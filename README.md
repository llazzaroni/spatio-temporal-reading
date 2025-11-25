# spatio-temporal-reading

A Python project for ***spatio-temporal reading analysis*** — preprocessing, exploring and plotting spatial-temporal data.

## Repository structure


├── data_preprocessing/ # scripts for cleaning / preparing raw data \\
├── exploration/ # exploratory analysis notebooks / scripts \\
├── plots/ # output visualizations (PNG, etc) \\
├── main.py # main entry point for the workflow \\
├── .gitignore \\
└── README.md


## ✅ Features

- Data ingestion and preprocessing (in the `data_preprocessing/` folder).  
- Exploratory analysis of spatial & temporal data (in `exploration/`).  
- Generation of visualisations (in `plots/`) showing temporal changes over space.  
- A unified `main.py` script to run the end-to-end workflow from raw data to visualization.

## 🚀 Getting Started

### Requirements

You’ll need Python (version 3.8+ recommended) and the following Python packages:

```bash
pip install numpy pandas matplotlib seaborn
