import argparse
import logging
from sched import scheduler
logging.basicConfig(level=logging.INFO)
import os

import random
import time

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin, DDPPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment
import torch
import debugger

from openfold.config import model_config
from openfold.data.data_modules import (
    OpenFoldDataModule,
    DummyDataLoader,
)
from openfold.model.model import AlphaFold
from openfold.model.torchscript import script_preset_
from openfold.np import residue_constants
from openfold.utils.argparse import remove_arguments
from openfold.utils.callbacks import (
    EarlyStoppingVerbose,
)
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.loss import AlphaFoldLoss, lddt_ca, compute_drmsd
from openfold.utils.lr_schedulers import AlphaFoldLRScheduler
from openfold.utils.seed import seed_everything
from openfold.utils.superimposition import superimpose
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.validation_metrics import (
    gdt_ts,
    gdt_ha,
)
from scripts.zero_to_fp32 import (
    get_fp32_state_dict_from_zero_checkpoint
)
# from scripts.zero_to_fp32_issue1422 import (
#     get_fp32_state_dict_from_zero_checkpoint
# )

from openfold.utils.logger import PerformanceLoggingCallback


class OpenFoldWrapper(pl.LightningModule):
    def __init__(self, config):
        super(OpenFoldWrapper, self).__init__()
        self.config = config
        self.model = AlphaFold(config)
        self.loss = AlphaFoldLoss(config.loss)
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )
        
        self.cached_weights = None

    def forward(self, batch):
        return self.model(batch)

    def _log(self, loss_breakdown, batch, outputs, train=True):
        phase = "train" if train else "val"
        for loss_name, indiv_loss in loss_breakdown.items():
            self.log(
                f"{phase}/{loss_name}", 
                indiv_loss, 
                on_step=train, on_epoch=(not train), logger=True,
            )

            if(train):
                self.log(
                    f"{phase}/{loss_name}_epoch",
                    indiv_loss,
                    on_step=False, on_epoch=True, logger=True,
                )

        with torch.no_grad():
            other_metrics = self._compute_validation_metrics(
                batch, 
                outputs,
                superimposition_metrics=(not train)
            )

        for k,v in other_metrics.items():
            self.log(
                f"{phase}/{k}", 
                v, 
                on_step=False, on_epoch=True, logger=True
            )

    def training_step(self, batch, batch_idx):
        if(self.ema.device != batch["aatype"].device):
            self.ema.to(batch["aatype"].device)

        # Run the model
        outputs = self(batch)
        
        # Remove the recycling dimension
        batch = tensor_tree_map(lambda t: t[..., -1], batch)
        # Compute loss
        loss, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        # Log it
        self._log(loss_breakdown, batch, outputs)
        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if(self.cached_weights is None):
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling 
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.cached_weights = tensor_tree_map(clone_param, self.model.state_dict())
            self.model.load_state_dict(self.ema.state_dict()["params"])

        # Run the model
        outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Compute loss and other metrics
        batch["use_clamped_fape"] = 0.
        _, loss_breakdown = self.loss(
            outputs, batch, _return_breakdown=True
        )

        self._log(loss_breakdown, batch, outputs, train=False)
        
    def validation_epoch_end(self, _):
        # Restore the model weights to normal
        self.model.load_state_dict(self.cached_weights)
        self.cached_weights = None

    def _compute_validation_metrics(self, 
        batch, 
        outputs, 
        superimposition_metrics=False
    ):
        metrics = {}
        
        gt_coords = batch["all_atom_positions"] # [*, N, 37, 3]
        pred_coords = outputs["final_atom_positions"] # [*, N, 37, 3]
        all_atom_mask = batch["all_atom_mask"] # [*, N, 37]
        # This is super janky for superimposition. Fix later
        gt_coords_masked = gt_coords * all_atom_mask[..., None] # [*, N, 37, 3]
        pred_coords_masked = pred_coords * all_atom_mask[..., None] # [*, N, 37, 3]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_masked_ca = gt_coords_masked[..., ca_pos, :] # [*, N, 3]
        pred_coords_masked_ca = pred_coords_masked[..., ca_pos, :] # [*, N, 3]
        all_atom_mask_ca = all_atom_mask[..., ca_pos] # [*, N]
    
        lddt_ca_score = lddt_ca(
            pred_coords,
            gt_coords,
            all_atom_mask,
            eps=self.config.globals.eps,
            per_residue=False,
        ) # [*]

        metrics["lddt_ca"] = lddt_ca_score

        drmsd_ca_score = compute_drmsd(
            pred_coords_masked_ca,
            gt_coords_masked_ca,
            mask=all_atom_mask_ca,
        ) # [*]

        metrics["drmsd_ca"] = drmsd_ca_score

        if(superimposition_metrics):
            superimposed_pred, _ = superimpose(
                gt_coords_masked_ca, pred_coords_masked_ca
            ) # [*, N, 3]
            gdt_ts_score = gdt_ts(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )
            gdt_ha_score = gdt_ha(
                superimposed_pred, gt_coords_masked_ca, all_atom_mask_ca
            )

            metrics["gdt_ts"] = gdt_ts_score
            metrics["gdt_ha"] = gdt_ha_score
    
        return metrics

    def configure_optimizers(self) -> torch.optim.Adam:
        optim_config = self.config.optimizer
        scheduler_config = self.config.scheduler
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=optim_config.lr,
            eps=optim_config.eps,
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
                # The unit of the scheduler's step size.
                # 'epoch' updates the scheduler on epoch end
                # whereas 'step' updates it after a optimizer update.
                "interval": "step",
                # How many epochs/steps should pass between calls to
                # `scheduler.step()`. 1 corresponds to updating the learning
                # rate after every epoch/step.
                "frequency": 1,
                "name": "AlphaFoldLRScheduler",
            }
        }

    def on_load_checkpoint(self, checkpoint):
        self.ema.load_state_dict(checkpoint["ema"])

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()


def main(args):
    if(args.seed is not None):
        seed_everything(args.seed) 

    config = model_config(
        args.config_preset, 
        train=True, 
        low_prec=(args.precision == 16),
    )
    model_module = OpenFoldWrapper(config)
    if(args.resume_from_ckpt and args.resume_model_weights_only):
        sd = get_fp32_state_dict_from_zero_checkpoint(args.resume_from_ckpt)
        sd = {k[len("module."):]:v for k,v in sd.items()}
        model_module.load_state_dict(sd)
        logging.info("Successfully loaded model weights...")

    # TorchScript components of the model
    if(args.script_modules):
        script_preset_(model_module)

    #data_module = DummyDataLoader("new_batch.pickle")
    data_module = OpenFoldDataModule(
        config=config.data, 
        batch_seed=args.seed,
        **vars(args)
    )

    data_module.prepare_data()
    data_module.setup()
    
    callbacks = []
    if(args.checkpoint_every_epoch):
        mc = ModelCheckpoint(
            filename="epoch{epoch:02d}-step{step}-val_loss={val/loss:.2f}",
            auto_insert_metric_name=False,
            monitor="val/loss",
            mode="min",
            every_n_epochs=1,
            save_last=False,
            save_top_k=10,
        )
        callbacks.append(mc)

    if(args.early_stopping):
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

    if(args.log_performance):
        global_batch_size = args.num_nodes * args.gpus
        perf = PerformanceLoggingCallback(
            log_file=os.path.join(args.output_dir, "performance_log.json"),
            global_batch_size=global_batch_size,
        )
        callbacks.append(perf)

    if(args.log_lr):
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    loggers = []
    if(args.wandb):
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

    if(args.deepspeed_config_path is not None):
        strategy = DeepSpeedPlugin(
            config=args.deepspeed_config_path,
        )
        if(args.wandb):
            wdb_logger.experiment.save(args.deepspeed_config_path)
            wdb_logger.experiment.save("openfold/config.py")
    elif (args.gpus is not None and args.gpus > 1) or args.num_nodes > 1:
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

    if(args.resume_model_weights_only):
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_data_dir", type=str,
        help="Directory containing training mmCIF files"
    )
    parser.add_argument(
        "train_alignment_dir", type=str,
        help="Directory containing precomputed training alignments"
    )
    parser.add_argument(
        "template_mmcif_dir", type=str,
        help="Directory containing mmCIF files to search for templates"
    )
    parser.add_argument(
        "output_dir", type=str,
        help=(
            "Directory in which to output checkpoints, logs, etc. Ignored "
            "if not on rank 0"
        )
    )
    parser.add_argument(
        "max_template_date", type=str,
        help=(
            "Cutoff for all templates. In training mode, templates are also "
            "filtered by the release date of the target"
        )
    )
    parser.add_argument(
        "--sabdab_summary_file", type=str,
        help="Summary file from SAbDab. Used to extract ground truth resolution"
    )
    parser.add_argument(
        "--distillation_data_dir", type=str, default=None,
        help="Directory containing training PDB files"
    )
    parser.add_argument(
        "--distillation_alignment_dir", type=str, default=None,
        help="Directory containing precomputed distillation alignments"
    )
    parser.add_argument(
        "--val_data_dir", type=str, default=None,
        help="Directory containing validation mmCIF files"
    )
    parser.add_argument(
        "--val_alignment_dir", type=str, default=None,
        help="Directory containing precomputed validation alignments"
    )
    parser.add_argument(
        "--kalign_binary_path", type=str, default="/usr/bin/kalign",
        help="Path to the kalign binary"
    )
    parser.add_argument(
        "--train_mapping_path", type=str, default=None,
        help=(
            "Optional path to a .json file containing a mapping from "
            "consecutive numerical indices to sample names. Used to filter "
            "the training set"
        )
    )
    parser.add_argument(
        "--distillation_mapping_path", type=str, default=None,
        help="See --train_mapping_path"
    )
    parser.add_argument(
        "--obsolete_pdbs_file_path", type=str, default=None,
        help=(
            "Path to obsolete.dat file containing list of obsolete PDBs and "
            "their replacements"
        )
    )
    parser.add_argument(
        "--template_release_dates_cache_path", type=str, default=None,
        help="Output of scripts/generate_mmcif_cache.py run on template mmCIF files"
    )
    parser.add_argument(
        "--use_small_bfd", type=bool_type, default=False,
        help="Whether to use a reduced version of the BFD database"
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
        "--log_performance", type=bool_type, default=False,
        help="Measure performance"
    )
    parser.add_argument(
        "--log_lr", type=bool_type, default=True,
        help="Whether to log the actual learning rate"
    )
    parser.add_argument(
        "--wandb", type=bool_type, default=False,
        help="Whether to log metrics to Weights & Biases"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None,
        help="wandb username or team name to which runs are attributed"
    )
    parser.add_argument(
        "--wandb_version", type=str, default=None,
        help="Sets the version, mainly used to resume a previous run."
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="Name of the wandb project to which this run will belong"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Name of the current experiment. Used for wandb logging"
    )
    parser.add_argument(
        "--script_modules", type=bool_type, default=False,
        help="Whether to TorchScript eligible components of the model"
    )
    parser.add_argument(
        "--train_embedding_dir", type=str, default=None,
        help="Directory containing pre-trained training residue embeddings"
    )
    parser.add_argument(
        "--distillation_embedding_dir", type=str, default=None,
        help="Directory containing pre-trained distillation residue embeddings"
    )
    parser.add_argument(
        "--val_embedding_dir", type=str, default=None,
        help="Directory containing pre-trained validation residue embeddings"
    )
    parser.add_argument(
        "--train_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--distillation_chain_data_cache_path", type=str, default=None,
    )
    parser.add_argument(
        "--config_preset", type=str, default="initial_training",
        help=(
            "Config setting. Choose e.g. 'initial_training', 'finetuning', "
            "'model_1', etc. By default, the actual values in the config are "
            "used."
        )
    )
    parser.add_argument(
        "--_alignment_index_path", type=str, default=None,
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
            "--accelerator", 
            "--resume_from_checkpoint",
            "--reload_dataloaders_every_epoch",
            "--reload_dataloaders_every_n_epochs",
        ]
    ) 

    args = parser.parse_args()

    if(args.seed is None and 
        ((args.gpus is not None and args.gpus > 1) or 
         (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    # process wandb args
    if (args.wandb):
        if args.wandb_version is not None:
            args.wandb_version = f"{args.config_preset}-{args.wandb_version}"
        if args.experiment_name is None:
            args.experiment_name = args.wandb_version

    # This re-applies the training-time filters at the beginning of every epoch
    args.reload_dataloaders_every_n_epochs = 1

    main(args)