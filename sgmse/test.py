import argparse
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

import os



if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     parser.add_argument("--middle_concat_attention", type=bool, default=False)  
     args = parser.parse_args()
     print(args.middle_concat_attention, type(args.middle_concat_attention))
     if args.middle_concat_attention:
         print("yes it is True")