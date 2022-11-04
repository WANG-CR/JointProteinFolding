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
from openfold.model.model import AlphaFold
from openfold.model.model_inv import AlphaFoldInverse
from openfold.np import residue_constants
from openfold.utils.argparse import remove_arguments
from openfold.utils.callbacks import EarlyStoppingVerbose
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.loss import AlphaFoldLoss, distogram_loss, lddt_ca, compute_drmsd, InverseLoss
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
        self.f_model = AlphaFold(config)
        self.g_model = AlphaFoldInverse(config)
        self.f_loss = AlphaFoldLoss(config.loss)
        self.g_loss = InverseLoss(config.loss)
        self.f_ema = ExponentialMovingAverage(
            model=self.f_model, decay=config.ema.decay
        )
        self.g_ema = ExponentialMovingAverage(
            model=self.g_model, decay=config.ema.decay
        )

        self.cached_weights = None

    def forward(self, batch):
        return self.f_model(batch), self.g_model(batch)

    def forward_joint(self, batch):
        f_outputs = self.f_model(batch)
        # we should consider the mask
        coords = f_outputs["final_atom_positions"] # [*, N, 37, 3]
        n_pos = residue_constants.atom_order["N"]
        gt_coords_n = coords[..., n_pos, :].unsqueeze(-2) # [*, N, 1, 3]
        ca_pos = residue_constants.atom_order["CA"]
        gt_coords_ca = coords[..., ca_pos, :].unsqueeze(-2) # [*, N, 3]
        c_pos = residue_constants.atom_order["C"]
        gt_coords_c = coords[..., c_pos, :].unsqueeze(-2) # [*, N, 3]
        o_pos = residue_constants.atom_order["O"]
        gt_coords_o = coords[..., o_pos, :].unsqueeze(-2) # [*, N, 3]
        coords_feats = torch.cat((gt_coords_n, gt_coords_ca, gt_coords_c, gt_coords_o), dim=-2)

        h_outputs = self.g_model.forward_h(batch, coords_feats)
        return f_outputs, h_outputs

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
        if(self.f_ema.device != batch["a"]["aatype"].device):
            self.f_ema.to(batch["a"]["aatype"].device)
            self.g_ema.to(batch["a"]["aatype"].device)

        if batch_idx % 2 == 0:
        # Run the model
            f_outputs, g_outputs = self(batch["a"])

            # Remove the recycling dimension
            batch_a = tensor_tree_map(lambda t: t[..., -1], batch["a"])

            # Compute f loss
            f_loss, f_loss_breakdown = self.f_loss(
                f_outputs, batch_a, _return_breakdown=True
            )
            # Log it
            self._log(f_loss_breakdown, batch_a, f_outputs)

            # Compute g loss
            logits = g_outputs["sm"]["seqs_logits"][-1]
            aatype = batch_a["aatype"]
            
            masked_pred = logits.masked_select(batch_a["seq_mask"].unsqueeze(-1).to(torch.bool)).view(-1, residue_constants.restype_num + 1) # Nl x 21
            masked_target = aatype.masked_select(batch_a["seq_mask"].to(torch.bool)).view(-1)    # Nl

            ce = F.cross_entropy(masked_pred, masked_target.long())
            ppl = ce.exp()
            dummy_loss = sum([v.float().sum() for v in g_outputs["sm"].values()])    # calculate other loss (pl distributed training)
            # Log it
            self.log('train/g_loss', ce, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train/g_PPL', ppl, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('train/g_loss_epoch', ce, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log('train/g_PPL_epoch', ppl, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

            g_loss = ce + 0. * dummy_loss
            loss = f_loss + g_loss
            return loss


        elif batch_idx % 2 == 1:
            f_outputs, h_outputs = self.forward_joint(batch["b"])
            plddt = f_outputs["plddt"][-1].mean()
            plddt_loss = -plddt/100 + 1.0
            # use fake loss, to avoid non-used parameters
            # these parameters will cause backpropagation error during DDP
            logits=f_outputs["sm"]["seqs_logits"]
            # [8, 1, 256, 21]
            distogram = f_outputs["distogram_logits"]
            # [1, 256, 256, 64]
            logits_loss = logits.sum()
            distogram_loss = distogram.sum()

            batch_b = tensor_tree_map(lambda t: t[..., -1], batch["b"])
            # Compute h loss
            logits_h = h_outputs["sm"]["seqs_logits"][-1]
            aatype = batch_b["aatype"]
            masked_target = aatype.masked_select(batch_b["seq_mask"].to(torch.bool)).view(-1)    # Nl

            masked_pred_h = logits_h.masked_select(batch_b["seq_mask"].unsqueeze(-1).to(torch.bool)).view(-1, residue_constants.restype_num + 1) # Nl x 21

            ce_h = F.cross_entropy(masked_pred_h, masked_target.long())
            ppl_h = ce_h.exp()
            dummy_loss_h = sum([v.float().sum() for v in h_outputs["sm"].values()])    # calculate other loss (pl distributed training)
            # Log it
            self.log('train/h_loss', ce_h, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log('train/h_PPL', ppl_h, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('train/h_loss_epoch', ce_h, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log('train/h_PPL_epoch', ppl_h, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            h_loss = ce_h + 0. * dummy_loss_h + plddt_loss + 0. * logits_loss + 0. * distogram_loss
            return h_loss

        

    def on_before_zero_grad(self, *args, **kwargs):
        self.f_ema.update(self.f_model)
        self.g_ema.update(self.g_model)

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if(self.cached_weights is None):
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling 
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.f_cached_weights = tensor_tree_map(clone_param, self.f_model.state_dict())
            self.f_model.load_state_dict(self.f_ema.state_dict()["params"])
            self.g_cached_weights = tensor_tree_map(clone_param, self.g_model.state_dict())
            self.g_model.load_state_dict(self.g_ema.state_dict()["params"])

        # Run the model
        f_outputs, g_outputs = self(batch)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # Compute loss and other metrics
        batch["use_clamped_fape"] = 0.
        f_loss, f_loss_breakdown = self.f_loss(
            f_outputs, batch, _return_breakdown=True
        )
        self._log(f_loss_breakdown, batch, f_outputs, train=False)


        # compute g loss
        logits = g_outputs["sm"]["seqs_logits"][-1]
        aatype = batch["aatype"]
        # only last step computed as ce
        masked_pred = logits.masked_select(batch["seq_mask"].unsqueeze(-1).to(torch.bool)).view(-1, residue_constants.restype_num+1) # Nl x 21
        masked_target = aatype.masked_select(batch["seq_mask"].to(torch.bool)).view(-1)    # Nl
        ce = F.cross_entropy(masked_pred, masked_target)
        ppl = ce.exp()
        
        logits[..., -1] = -9999 # zero out UNK.
        sampled_seqs = logits.argmax(dim=-1)    # greedy sampling
        masked_sampled_seqs = sampled_seqs.masked_select(batch["seq_mask"].to(torch.bool)).view(-1) # N x Nl
        aars = masked_sampled_seqs.eq(masked_target).float().mean()

        self.log('val/g_loss', ce, on_step=False, on_epoch=True, prog_bar=False, logger=False, sync_dist=True)
        self.log('val/g_PPL', ppl, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)    # show
        self.log('val/g_AAR', aars, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)  



        # # Compute h loss
        # logits_h = h_outputs["sm"]["seqs_logits"][-1]

        # masked_pred_h = logits_h.masked_select(batch["seq_mask"].unsqueeze(-1).to(torch.bool)).view(-1, residue_constants.restype_num + 1) # Nl x 21

        # ce_h = F.cross_entropy(masked_pred_h, masked_target)
        # ppl_h = ce_h.exp()
        # logits_h[..., -1] = -9999 # zero out UNK.
        # sampled_seqs_h = logits_h.argmax(dim=-1)    # greedy sampling
        # masked_sampled_seqs_h = sampled_seqs_h.masked_select(batch["seq_mask"].to(torch.bool)).view(-1) # N x Nl
        # aars_h = masked_sampled_seqs_h.eq(masked_target).float().mean()

        # # Log it
        # self.log('val/h_loss', ce_h, on_step=False, on_epoch=True, prog_bar=False, logger=False, sync_dist=True)
        # self.log('val/h_PPL', ppl_h, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)    # show
        # self.log('val/h_AAR', aars_h, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)  

        # loss = ce + f_loss
        # self.log(
        #         f"val/loss", 
        #         loss, 
        #         on_step=False, on_epoch=True, logger=True, sync_dist=True
        #     )

        
    def validation_epoch_end(self, _):
        # Restore the model weights to normal
        self.f_model.load_state_dict(self.f_cached_weights)
        self.f_cached_weights = None

        self.g_model.load_state_dict(self.g_cached_weights)
        self.g_cached_weights = None

    def _compute_validation_metrics(self, 
        batch, 
        outputs, 
        superimposition_metrics=False
    ):
        metrics = {}
        
        gt_coords = batch["all_atom_positions"].float() # [*, N, 37, 3]
        pred_coords = outputs["final_atom_positions"].float() # [*, N, 37, 3]
        all_atom_mask = batch["all_atom_mask"].float() # [*, N, 37]
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
            [{"params":self.f_model.parameters()},{"params":self.g_model.parameters()}],
            lr=optim_config.lr,
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
        self.f_ema.load_state_dict(checkpoint["f_ema"])
        self.g_ema.load_state_dict(checkpoint["g_ema"])

    def on_save_checkpoint(self, checkpoint):
        checkpoint["f_ema"] = self.f_ema.state_dict()
        checkpoint["g_ema"] = self.g_ema.state_dict()


def main(args):
    if(args.seed is not None):
        seed_everything(args.seed) 

    logging.info(f"args.is_antibody is {args.is_antibody}")
    config = model_config(
        name=args.config_preset,
        yaml_config_preset=args.yaml_config_preset,
        train=True, 
        low_prec=(args.precision == 16),
    )
    model_module = OpenFoldWrapper(config)
    # if(args.resume_from_ckpt and args.resume_model_weights_only):
    #     sd = get_fp32_state_dict_from_zero_checkpoint(args.resume_from_ckpt)
    #     sd = {k[len("module."):]:v for k,v in sd.items()}
    #     model_module.load_state_dict(sd)
    #     logging.info("Successfully loaded model weights...")

    logging.info(f"args.resume_model_weights_only is {args.resume_model_weights_only}")
    logging.info(f"args.resume_from_ckpt_f is {args.resume_from_ckpt_f}")
    if(args.resume_model_weights_only):
        if args.resume_from_ckpt_f:
            # sd = get_fp32_state_dict_from_zero_checkpoint(args.resume_from_ckpt_f)
            sd = torch.load(args.resume_from_ckpt_f)
            dict1 = {k[len("module."):]:v for k,v in sd.items()}
            for key, values in dict1.items():
                print(f"key is {key}")
            sd = {k[len("module."):]:v for k,v in sd.items()}
            model_module.load_state_dict(sd)
            logging.info("Successfully loaded model_f weights...")

    parallel_data_module = OpenFoldDataModule(
        config=config.data, 
        batch_seed=args.seed,
        **vars(args)
    )

    parallel_data_module.prepare_data()
    parallel_data_module.setup()

    # process fasta file
    sequence_data_module = OpenFoldDataModule(
        config=config.data, 
        batch_seed=args.seed,
        train_data_dir=args.fasta_dir,
        train_epoch_len=args.train_epoch_len,
        is_antibody=args.is_antibody,
    )
    sequence_data_module.prepare_data()
    sequence_data_module.setup()

    callbacks = []
    if(args.checkpoint_every_epoch):
        mc = ModelCheckpoint(
            filename="epoch{epoch:02d}-step{step}-val_loss={val/loss:.3f}",
            auto_insert_metric_name=False,
            monitor="val/loss",
            mode="min",
            every_n_epochs=1,
            save_last=False,
            save_top_k=30,
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
            if args.yaml_config_preset is not None:
                wdb_logger.experiment.save(args.yaml_config_preset)
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

    # multi data module training
    train_dataloader={"a": parallel_data_module.train_dataloader(), "b": sequence_data_module.train_dataloader()}
    trainer.fit(
        model_module, 
        train_dataloaders=train_dataloader,
        val_dataloaders=parallel_data_module.val_dataloader(),
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
        help="Directory containing training pdb files"
    )
    parser.add_argument(
        "--fasta_dir", type=str, default='None',
        help="Directory containing training fasta files"
    )
    parser.add_argument(
        "--output_dir", type=str, default='invfold_outputs',
        help=(
            "Directory in which to output checkpoints, logs, etc. Ignored "
            "if not on rank 0"
        )
    )
    parser.add_argument(
        "--is_antibody", type=bool, default=False,
        help="training on antibody or not"
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
        "--resume_from_ckpt_f", type=str, default=None,
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
        "--config_preset", type=str, default=None,
        help=(
            "Config setting. Choose e.g. 'initial_training', 'finetuning', "
            "'model_1', etc. By default, the actual values in the config are "
            "used."
        )
    )
    parser.add_argument(
        "--yaml_config_preset", type=str, default=None,
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
    if(args.wandb):
        if args.wandb_version is not None:
            args.wandb_version = f"{args.config_preset}-{args.wandb_version}"
        if args.experiment_name is None:
            args.experiment_name = args.wandb_version

    logging.info(f"args train data dir is {args.train_data_dir}")
    logging.info(f"args yaml config is {args.yaml_config_preset}")

    # This re-applies the training-time filters at the beginning of every epoch
    args.reload_dataloaders_every_n_epochs = 1

    main(args)
