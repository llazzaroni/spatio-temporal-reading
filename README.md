# spatio-temporal-reading

A Python project for ***spatio-temporal reading analysis*** — preprocessing, exploring and plotting spatial-temporal data.


## Features

- Data preprocessing: include the indices in the texts corresponding to each fixation. The data preprocessing code expects the already processed meco_df dataset as obtained in https://github.com/rycolab/spatio-temporal-reading. To run the data preprocessing:
```bash
python main.py --include-indices True --data "DATA_DIR"
```
- Exploratory analysis of spatial & temporal data. To obtain the plots run:
```bash
python main.py --make-plots True --data "DATA_DIR" --output "OUTPUT_DIR"
``` 