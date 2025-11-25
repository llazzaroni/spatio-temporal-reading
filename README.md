# spatio-temporal-reading

A Python project for ***spatio-temporal reading analysis*** — preprocessing, exploring and plotting spatial-temporal data.


## Features

- Data preprocessing: include the indices in the texts corresponding to each fixation. To run the data preprocessing:
```bash
python main.py --include-indices True --data "DATA_DIR"
```
- Exploratory analysis of spatial & temporal data (in `exploration/`).  
- Generation of visualisations (in `plots/`) showing temporal changes over space.  
- A unified `main.py` script to run the end-to-end workflow from raw data to visualization.

## 🚀 Getting Started

### Requirements

You’ll need Python (version 3.8+ recommended) and the following Python packages:

```bash
pip install numpy pandas matplotlib seaborn
