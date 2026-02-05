# spatio-temporal-reading

A Python project for ***spatio-temporal reading analysis*** — preprocessing, exploring and plotting spatial-temporal data.


## Features

- Data preprocessing: include the indices in the texts corresponding to each fixation. The data preprocessing code expects the already processed meco_df dataset as obtained in https://github.com/rycolab/spatio-temporal-reading. To run the data preprocessing:
```bash
python main.py --include-indices --data DATA_DIR
```
- Exploratory analysis of spatial & temporal data. To obtain the plots run:
```bash
python main.py --make-plots --data DATA_DIR --output OUTPUT_DIR
``` 

- Training and visualization of a simple model to predict the positions of the next fixations and saccade intervals. To train the model:

```bash
python main.py --train --data DATA_DIR --output OUTPUT_DIR
```

Optionally, add the flags --n-layers, --d-model, --n-components, --epochs to, respectively, control the number of layers and dimensions of the hidden state of the transformer, the number of components of the final distribution and the number of training epochs.

To run the visualization of the predictions of the model, run:

```bash
python main.py --visualize-model --data DATA_DIR --output OUTPUT_DIR --checkpoint-path CHECKPOINT_PATH
```

This command will produce four gifs, showcasing the predictions of the model vs actual points for fixation locations and saccade durations, for a sequence in the training set and one in the validation set. Note that, to be able to run the last command, the model has to be trained first and the checkpoint path has to point to the location of the .pt file containing the model. Additionally, the flags --train-index and --val-index can be chosen.

To train the baseline models, run:

```bash
python main.py --train-baseline --data DATA_DIR
```