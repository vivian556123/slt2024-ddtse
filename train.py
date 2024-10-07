import argparse
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import wandb

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module import SpecsDataModule
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel,DiscriminativeModel
from sgmse.conditional_model import ConditionalScoreModel, ConditionalDiscriminativeModel 
from sgmse.conditional_model import ConditionalDDTSEModel

import os
import deepspeed

def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups


if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
          parser_.add_argument("--no_wandb", action='store_true', help="Turn off logging to W&B, using local default logger instead")
          parser_.add_argument("--ddtse_save_dir", type=str,  default="logs")
          parser_.add_argument("--condition", type=str, choices=("no", "yes"), default="no", help="no for Spec, yes for ConditionalSpec")
          parser_.add_argument("--discriminatively", type=str, choices=("no", "yes"), default="no",  help="Train the backbone as a discriminative model instead")  
          parser_.add_argument("--algorithm_type", type=str, choices=("no","DDTSE","DDTSE_spkencoder"), default="DDTSE",  help="DDTSE or other algorithm")  
          parser_.add_argument("--use_2_channel",  action='store_true',  help="Use 2channels or 4 channels in DDTSE")  
          parser_.add_argument("--pretrained_model",   type=str, default = "no", help="pretrained model or resume from checkpoint")  
          parser_.add_argument("--deepspeed", action='store_true', help="Turn off deepspeed")


     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     parser = pl.Trainer.add_argparse_args(parser)
     ScoreModel.add_argparse_args(
          parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
     # Add backbone args
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     # Add data module args
     data_module_cls = SpecsDataModule
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
     # Parse args and separate into groups
     args = parser.parse_args()
     print(args)
     # if args.ddtse_save_dir does not exist
     if not os.path.exists(args.ddtse_save_dir):
          os.makedirs(args.ddtse_save_dir)
     with open (os.path.join(args.ddtse_save_dir, "args.txt"), "w") as f:
          f.write(str(args))
     arg_groups = get_argparse_groups(parser)

     # Initialize logger, trainer, model, datamodule
     if args.pretrained_model == "no":
          if args.algorithm_type == "DDTSE":
               print("use_2_channel",args.use_2_channel)
               model = ConditionalDDTSEModel(
                              backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
                              **{
                                   **vars(arg_groups['ScoreModel']),
                                   **vars(arg_groups['SDE']),
                                   **vars(arg_groups['Backbone']),
                                   **vars(arg_groups['DataModule']), "discriminative": args.use_2_channel
                              }
                         )
          else:
               if args.condition == "no" and args.discriminatively=="no":
                    model = ScoreModel(
                         backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
                         **{
                              **vars(arg_groups['ScoreModel']),
                              **vars(arg_groups['SDE']),
                              **vars(arg_groups['Backbone']),
                              **vars(arg_groups['DataModule'])
                         }
                    )
               elif args.condition == "no" and args.discriminatively=="yes":
                    model = DiscriminativeModel(
                         backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
                         **{
                              **vars(arg_groups['ScoreModel']),
                              **vars(arg_groups['SDE']),
                              **vars(arg_groups['Backbone']),
                              **vars(arg_groups['DataModule']),"discriminative": True
                         }
                    )
               elif args.condition == "yes"  and args.discriminatively=="no":
                    model = ConditionalScoreModel(
                         backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
                         **{
                              **vars(arg_groups['ScoreModel']),
                              **vars(arg_groups['SDE']),
                              **vars(arg_groups['Backbone']),
                              **vars(arg_groups['DataModule']), 
                         }
                    )
               elif args.condition == "yes"  and args.discriminatively=="yes":
                    model = ConditionalDiscriminativeModel(
                         backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
                         **{
                              **vars(arg_groups['ScoreModel']),
                              **vars(arg_groups['SDE']),
                              **vars(arg_groups['Backbone']),
                              **vars(arg_groups['DataModule']), "discriminative": True
                         }
                    )
     else: 
          checkpoint_file = args.pretrained_model
          if args.algorithm_type=="DDTSE":
               model = ConditionalDDTSEModel.load_from_checkpoint(checkpoint_file, backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
                              **{
                                   **vars(arg_groups['ScoreModel']),
                                   **vars(arg_groups['SDE']),
                                   **vars(arg_groups['Backbone']),
                                   **vars(arg_groups['DataModule']), "discriminative": args.use_2_channel
                              })
          else: 
               if args.condition == "no" and args.discriminatively == "no": 
                    model = ScoreModel.load_from_checkpoint(checkpoint_file, backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
                         **{
                              **vars(arg_groups['ScoreModel']),
                              **vars(arg_groups['SDE']),
                              **vars(arg_groups['Backbone']),
                              **vars(arg_groups['DataModule'])
                         })
               elif args.condition == "no" and args.discriminatively == "yes":
                    model = DiscriminativeModel.load_from_checkpoint(checkpoint_file,backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
                         **{
                              **vars(arg_groups['ScoreModel']),
                              **vars(arg_groups['SDE']),
                              **vars(arg_groups['Backbone']),
                              **vars(arg_groups['DataModule']),"discriminative": True
                         })
               elif args.condition == "yes" and args.discriminatively == "no":
                    model = ConditionalScoreModel.load_from_checkpoint(checkpoint_file, backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
                         **{
                              **vars(arg_groups['ScoreModel']),
                              **vars(arg_groups['SDE']),
                              **vars(arg_groups['Backbone']),
                              **vars(arg_groups['DataModule']), 
                         })
               elif args.condition == "yes" and args.discriminatively == "yes":
                    model = ConditionalDiscriminativeModel.load_from_checkpoint(checkpoint_file, backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
                         **{
                              **vars(arg_groups['ScoreModel']),
                              **vars(arg_groups['SDE']),
                              **vars(arg_groups['Backbone']),
                              **vars(arg_groups['DataModule']), "discriminative": True
                         })
          
     # Set up logger configuration     
     if args.no_wandb:
          logger = TensorBoardLogger(save_dir=args.ddtse_save_dir, name="tensorboard")
     else:
          logger = WandbLogger(project="sgmse", log_model=True, save_dir=args.ddtse_save_dir)
          logger.experiment.log_code(".")

     # Set up callbacks for logger
     callbacks = [ModelCheckpoint(dirpath=f"{args.ddtse_save_dir}/{logger.version}", save_last=True, filename='{epoch}-last')]
     if args.num_eval_files:
          checkpoint_callback_pesq = ModelCheckpoint(dirpath=f"{args.ddtse_save_dir}/{logger.version}", 
               save_top_k=10, monitor="pesq", mode="max", filename='{epoch}-{pesq:.2f}')
          checkpoint_callback_si_sdr = ModelCheckpoint(dirpath=f"{args.ddtse_save_dir}/{logger.version}", 
               save_top_k=10, monitor="si_sdr", mode="max", filename='{epoch}-{si_sdr:.2f}')
          callbacks += [checkpoint_callback_pesq, checkpoint_callback_si_sdr]

     # Initialize the Trainer and the DataModule
     if args.deepspeed:
          if args.discriminatively=="no":
               trainer = pl.Trainer.from_argparse_args(
                    arg_groups['pl.Trainer'],
                    strategy="deepspeed_stage_3_offload", logger=logger,
                    log_every_n_steps=10, num_sanity_val_steps=0,
                    callbacks=callbacks, precision=16
               )
          else:
               trainer = pl.Trainer.from_argparse_args(
                    arg_groups['pl.Trainer'],
                    strategy="deepspeed_stage_3_offload", logger=logger,
                    log_every_n_steps=10, num_sanity_val_steps=0,
                    callbacks=callbacks, precision=16
               )

     else: 
          if args.discriminatively=="no":
               trainer = pl.Trainer.from_argparse_args(
                    arg_groups['pl.Trainer'],
                    strategy=DDPPlugin(find_unused_parameters=True), logger=logger,
                    log_every_n_steps=10, num_sanity_val_steps=0,
                    callbacks=callbacks
               )
          else:
               trainer = pl.Trainer.from_argparse_args(
                    arg_groups['pl.Trainer'],
                    strategy=DDPPlugin(find_unused_parameters=True), logger=logger,
                    log_every_n_steps=10, num_sanity_val_steps=0,
                    callbacks=callbacks
               )

     # Train model
     trainer.fit(model)
     #trainer.validate(model)
