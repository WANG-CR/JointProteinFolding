import argparse
import logging
logging.basicConfig(level=logging.INFO)

import os
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin, DDPPlugin

from openfold.config import model_config
from openfold.data.data_modules import OpenFoldDataModule
from openfold.np.residue_constants import restype_num
from openfold.model.model import AlphaFold
from openfold.model.model_inv import AlphaFoldInverse
from openfold.np import residue_constants
from openfold.utils.argparse import remove_arguments
from openfold.utils.callbacks import EarlyStoppingVerbose
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.loss import AlphaFoldLoss, lddt_ca, compute_drmsd, seqs_loss, InverseLoss
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils.seed import seed_everything
from openfold.utils.superimposition import superimpose
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.validation_metrics import gdt_ts, gdt_ha
from openfold.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

import debugger


class OpenFoldWrapper(pl.LightningModule):
    def __init__(self, config):
        super(OpenFoldWrapper, self).__init__()
        self.config = config
        self.model = AlphaFoldInverse(config)
        # self.dummyloss = AlphaFoldLoss(config.loss)
        self.loss = InverseLoss(config.loss)
        self.ema_enabled = False
        if self.ema_enabled:
            self.ema = ExponentialMovingAverage(
                model=self.model, decay=config.ema.decay
            )
        self.cached_weights = None
    
    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        if self.ema_enabled:
            if (self.ema.device != batch["aatype"].device):
                self.ema.to(batch["aatype"].device)

        # Run the model
        outputs = self(batch)
        
        # only last step computed as ce
        logits = outputs["sm"]["seqs_logits"][-1]
        aatype = batch["aatype"][..., -1]
        
        # calculate ce and aar
        masked_pred = logits.masked_select(batch["seq_mask"][..., -1].unsqueeze(-1).to(torch.bool)).view(-1, restype_num + 1) # Nl x 21
        masked_target = aatype.masked_select(batch["seq_mask"][..., -1].to(torch.bool)).view(-1)    # Nl
        ce = F.cross_entropy(masked_pred, masked_target.long())
        ppl = ce.exp()

        sample_logits = logits.clone().detach()
        sample_logits[..., -1] = -9999 # zero out UNK.
        sampled_seqs = sample_logits.argmax(dim=-1)
        masked_sampled_seqs = sampled_seqs.masked_select(batch["seq_mask"][..., -1].to(torch.bool)).view(-1)
        aars = masked_sampled_seqs.eq(masked_target).float().mean()

        dummy_loss = sum([v.float().sum() for v in outputs["sm"].values()])    # calculate other loss (pl distributed training)

        # Log it
        self.log('train/g_loss', ce, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train/g_PPL', ppl, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('train/g_aar', aars, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train/g_loss_epoch', ce, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/g_PPL_epoch', ppl, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/g_aar_epoch', aars, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        loss = ce + 0. * dummy_loss
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.ema_enabled:
            self.ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if self.ema_enabled:
            if(self.cached_weights is None):
                # model.state_dict() contains references to model weights rather
                # than copies. Therefore, we need to clone them before calling 
                # load_state_dict().
                clone_param = lambda t: t.detach().clone()
                self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
                self.model.load_state_dict(self.ema.state_dict()["params"])

        outputs = self(batch)
        # batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # only last step computed as ce
        logits = outputs["sm"]["seqs_logits"][-1]
        aatype = batch["aatype"][..., -1]

        # only last step computed as ce
        masked_pred = logits.masked_select(batch["seq_mask"][..., -1].unsqueeze(-1).to(torch.bool)).view(-1, restype_num+1) # Nl x 21
        masked_target = aatype.masked_select(batch["seq_mask"][..., -1].to(torch.bool)).view(-1)    # Nl
        ce = F.cross_entropy(masked_pred, masked_target)
        ppl = ce.exp()
        
        logits[..., -1] = -9999 # zero out UNK.
        sampled_seqs = logits.argmax(dim=-1)    # greedy sampling
        masked_sampled_seqs = sampled_seqs.masked_select(batch["seq_mask"][..., -1].to(torch.bool)).view(-1) # N x Nl
        aars = masked_sampled_seqs.eq(masked_target).float().mean()

        self.log('val/g_loss', ce, on_step=False, on_epoch=True, prog_bar=False, logger=False, sync_dist=True)
        self.log('val/g_PPL', ppl, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)    # show
        self.log('val/g_AAR', aars, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)  
        self.log('val/loss', ce, on_step=False, on_epoch=True, prog_bar=False, logger=False, sync_dist=True)


    def validation_epoch_end(self, _):
        # Restore the model weights to normal
        if self.ema_enabled:
            self.model.load_state_dict(self.cached_weights)
            self.cached_weights = None


    def configure_optimizers(self) -> torch.optim.Adam:
        optim_config = self.config.optimizer

        # https://github.com/Lightning-AI/lightning/issues/5558
        scheduler_config = self.config.scheduler
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=optim_config.lr,
            # weight_decay=optim_config.weight_decay,
            eps=optim_config.eps,
            weight_decay=1e-6,
        )
        lr_scheduler = AlphaFoldLRScheduler(
            optimizer,
            max_lr=optim_config.lr,
            **scheduler_config,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "AlphaFoldLRScheduler",
            }
        }

    def on_load_checkpoint(self, checkpoint):
        if self.ema_enabled:
            self.ema.load_state_dict(checkpoint["ema"])

    def on_save_checkpoint(self, checkpoint):
        if self.ema_enabled:
            checkpoint["ema"] = self.ema.state_dict()


def main(args):
    if args.seed is not None:
        seed_everything(args.seed) 

    config = model_config(
        name=args.config_preset,
        yaml_config_preset=args.yaml_config_preset,
        train=True, 
        low_prec=(args.precision == 16),
    )
    model_module = OpenFoldWrapper(config)
    if args.resume_from_ckpt and args.resume_model_weights_only:
        sd = get_fp32_state_dict_from_zero_checkpoint(args.resume_from_ckpt)
        sd = {k[len("module."):]:v for k,v in sd.items()}
        model_module.load_state_dict(sd)
        logging.info("Successfully loaded model weights...")

    data_module = OpenFoldDataModule(
        config=config.data, 
        batch_seed=args.seed,
        **vars(args)
    )

    data_module.prepare_data()
    data_module.setup()
    
    callbacks = []
    if args.checkpoint_every_epoch:
        if args.deepspeed_config_path is not None or not args.wandb:
            dirpath = None
        else:
            dirpath = os.path.join(
                args.output_dir,
                args.wandb_project,
                args.wandb_version,
                "checkpoints",
            )
        mc = ModelCheckpoint(
            filename="epoch{epoch:02d}-step{step}-val_loss={val/loss:.3f}",
            dirpath=dirpath,
            auto_insert_metric_name=False,
            monitor="val/loss",
            mode="min",
            every_n_epochs=1,
            save_last=False,
            save_top_k=1,
        )
        callbacks.append(mc)

    if args.early_stopping:
        es = EarlyStoppingVerbose(
            monitor="val/loss",
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode="min",
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)

    if args.log_lr:
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    loggers = []
    if args.wandb:
        # https://docs.wandb.ai/ref/python/init
        wdb_logger = WandbLogger(
            name=args.experiment_name,
            save_dir=args.output_dir,
            version=args.wandb_version,
            project=args.wandb_project,
            offline=True,
            **{"entity": args.wandb_entity}
        )
        loggers.append(wdb_logger)
        wandb_log_dir = os.path.join(args.output_dir, "wandb")
        if not os.path.exists(wandb_log_dir):
            logging.info(f"generating directory for wandb logging located at {wandb_log_dir}")
            os.makedirs(wandb_log_dir, exist_ok=True)

    if args.deepspeed_config_path is not None:
        strategy = DeepSpeedPlugin(
            config=args.deepspeed_config_path,
        )
        if args.wandb:
            wdb_logger.experiment.save(args.deepspeed_config_path)
            wdb_logger.experiment.save("openfold/config.py")
            if args.yaml_config_preset is not None:
                wdb_logger.experiment.save(args.yaml_config_preset)
    elif (args.gpus is not None and args.gpus > 1) or (args.devices is not None and args.devices >1) or args.num_nodes > 1:
        strategy = DDPPlugin(find_unused_parameters=False)
    else:
        strategy = None
   
    trainer = pl.Trainer.from_argparse_args(
        args,
        default_root_dir=args.output_dir,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
    )

    if args.resume_model_weights_only:
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt

    trainer.fit(
        model_module, 
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )



def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "train_data_dir", type=str, default=None,
        help="Directory containing training mmCIF files"
    )
    parser.add_argument(
        "output_dir", type=str, default='invfold_outputs',
        help=(
            "Directory in which to output checkpoints, logs, etc. Ignored "
            "if not on rank 0"
        )
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--early_stopping", type=bool_type, default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help=(
            "The smallest decrease in validation loss that counts as an "
            "improvement for the purposes of early stopping"
        )
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
    )
    parser.add_argument(
        "--train_epoch_len", type=int, default=None,
        help=(
            "The virtual length of each training epoch. Stochastic filtering "
            "of training data means that training datasets have no "
            "well-defined length. This virtual length affects frequency of "
            "validation & checkpointing (by default, one of each per epoch)."
            "If set to None, use the length of the dataset as epoch_len."
        )
    )
    parser.add_argument(
        "--checkpoint_every_epoch", type=bool_type, default=True,
        help="Whether to checkpoint at the end of every training epoch"
    )
    parser.add_argument(
        "--log_lr", type=bool_type, default=True,
        help="Whether to log the actual learning rate"
    )
    parser.add_argument(
        "--wandb", type=bool_type, default=True,
        help="Whether to log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="chuanrui",
        help="wandb username or team name to which runs are attributed"
    )
    parser.add_argument(
        "--wandb_version", type=str, default="invfold_init",
        help="Sets the version, mainly used to resume a previous run."
    )
    parser.add_argument(
        "--wandb_project", type=str, default="invfolding",
        help="Name of the wandb project to which this run will belong"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Name of the current experiment. Used for wandb logging"
    )
    parser.add_argument(
        "--config_preset", type=str, default=None,
        help=(
            "Config setting. Choose e.g. 'initial_training', 'finetuning', "
            "'model_1', etc. By default, the actual values in the config are "
            "used."
        )
    )
    parser.add_argument(
        "--yaml_config_preset", type=str, default="yaml_config/inverse.yml",
        help=(
            "A path to a yaml file that contains the updated config setting. "
            "If it is set, the config_preset will be overwrriten as the basename "
            "of the yaml_config_preset."
        )
    )
    parser = pl.Trainer.add_argparse_args(parser)
   
    # Disable the initial validation pass
    parser.set_defaults(
        num_sanity_val_steps=0,
    )

    # Remove some buggy/redundant arguments introduced by the Trainer
    remove_arguments(
        parser, 
        [
            "--devices", 
            "--num_nodes", 
            "--accelerator", 
            "--resume_from_checkpoint",
            "--reload_dataloaders_every_epoch",
            "--reload_dataloaders_every_n_epochs",
        ]
    ) 

    parser.add_argument(
        "--accelerator", type=str, default=None,
        help=(
            "specify the devices among 'cpu', 'gpu', 'auto'"
        )
    )
    parser.add_argument(
        "--devices", type=int, default=None,
        help=(
            "number of process per node"
        )
    )
    parser.add_argument(
        "--num_nodes", type=int, default=1,
        help=(
            "number of nodes"
        )
    )
    
    args = parser.parse_args()

    if(args.seed is None and 
        ((args.gpus is not None and args.gpus > 1) or 
         (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    if(args.config_preset is None and args.yaml_config_preset is None):
        raise ValueError(
            "Either --config_preset or --yaml_config_preset should be specified."
        )

    if(args.yaml_config_preset is not None):
        if not os.path.exists(args.yaml_config_preset):
            raise FileNotFoundError(f"{os.path.abspath(args.yaml_config_preset)}")
        args.config_preset = os.path.splitext(
            os.path.basename(args.yaml_config_preset)
        )[0]
        logging.info(f"the config_preset is set as {args.config_preset} by yaml_config_preset.")

    # process wandb args
    if args.wandb:
        if args.wandb_version is not None:
            args.wandb_version = f"{args.config_preset}-{args.wandb_version}"
        if args.experiment_name is None:
            args.experiment_name = args.wandb_version

    logging.info(f"args train data dir is {args.train_data_dir}")
    logging.info(f"args yaml config is {args.yaml_config_preset}")
    # This re-applies the training-time filters at the beginning of every epoch
    args.reload_dataloaders_every_n_epochs = 1

    main(args)