from spatio_temporal_reading.data.data import MecoDataset

def main(datapath):

    train_ds = MecoDataset(mode="train", datadir=datapath)
    #val_ds = MecoDataset(mode="val", datadir=datapath)
    #print(train_ds.__get_item__(index=300).shape)
