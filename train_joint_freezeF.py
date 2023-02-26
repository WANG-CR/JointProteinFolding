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
from openfold.model.esm.esmfold import ESMFold, ESMFoldConfig, constructConfigFromYAML
import debugger


class OpenFoldWrapper(pl.LightningModule):
    def __init__(self, config):
        super(OpenFoldWrapper, self).__init__()
        self.config = config
        model_data = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_3B_v1.pt", progress=False, map_location="cpu")
        cfg = constructConfigFromYAML(config)
        model_state = model_data["model"]
        self.f_model = ESMFold(esmfold_config=cfg, using_fair=True)
        self.f_model.load_state_dict(model_state, strict=False)
        self.f_model.requires_grad_(False)
        self.g_model = AlphaFoldInverse(config)
        self.g_loss = InverseLoss(config.loss)
        self.g_ema = ExponentialMovingAverage(
            model=self.g_model, decay=config.ema.decay
        )

        self.cached_weights = None

    def forward(self, batch):
        return self.f_model(batch), self.g_model(batch)

    def forward_joint(self, batch, sequence):
        # self.f_model.eval()
        with torch.no_grad():
            f_outputs = self.f_model.infer_bbs(sequence, num_recycles=3, cpu_only=True)
        # logging.info(f">>> debug: input sequence shape is {sequence.shape}")
        logging.info(f">>> debug: predicted coords_bb shape is {f_outputs['bb_coords'].shape}")
        logging.info(f">>> debug: ground_truth aatype shape is {f_outputs['aatype'].shape}")
        bb_feats = f_outputs['bb_coords']
        h_outputs = self.g_model.forward_h(batch, bb_feats)
        return h_outputs

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
        if(self.g_ema.device != batch["aatype"].device):
            self.g_ema.to(batch["aatype"].device)

        # sequence = residue_constants.aatype_to_sequence(batch["aatype"][0, ..., -1])
        # logging.info(f">>> debug: input sequence 1 is {batch['aatype'][0, ..., -1]}")
        # logging.info(f">>> debug: input sequence 2 is {batch['aatype'][1, ..., -1]}")
        sequence = residue_constants.aatypeList_to_sequence(batch["aatype"][..., -1].tolist())
        h_outputs = self.forward_joint(batch, sequence)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)
        logits_h = h_outputs["sm"]["seqs_logits"][-1]
        aatype = batch["aatype"]
        masked_target = aatype.masked_select(batch["seq_mask"].to(torch.bool)).view(-1)    # Nl
        masked_pred_h = logits_h.masked_select(batch["seq_mask"].unsqueeze(-1).to(torch.bool)).view(-1, residue_constants.restype_num + 1) # Nl x 21
        
        logits = logits_h.clone().detach()
        logits[..., -1] = -9999 # zero out UNK.
        sampled_seqs = logits.argmax(dim=-1)
        # logging.info(f">>> samples sequence shape is {sampled_seqs.shape}")
        masked_sampled_seqs = sampled_seqs.masked_select(batch["seq_mask"].to(torch.bool)).view(-1)
        aars = masked_sampled_seqs.eq(masked_target).float().mean()

        ce_h = F.cross_entropy(masked_pred_h, masked_target.long())
        ppl_h = ce_h.exp()
        dummy_loss_h = sum([v.float().sum() for v in h_outputs["sm"].values()])    # calculate other loss (pl distributed training)
        # Log it
        self.log('train/h_loss', ce_h, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('train/h_PPL', ppl_h, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train/h_aar', aars, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train/h_loss_epoch', ce_h, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/h_PPL_epoch', ppl_h, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/h_aar_epoch', aars, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        h_loss = ce_h + 0. * dummy_loss_h
        return h_loss

    def on_before_zero_grad(self, *args, **kwargs):
        self.g_ema.update(self.g_model)

    def validation_step(self, batch, batch_idx):
        # At the start of validation, load the EMA weights
        if(self.cached_weights is None):
            # model.state_dict() contains references to model weights rather
            # than copies. Therefore, we need to clone them before calling 
            # load_state_dict().
            clone_param = lambda t: t.detach().clone()
            self.g_cached_weights = tensor_tree_map(clone_param, self.g_model.state_dict())
            self.g_model.load_state_dict(self.g_ema.state_dict()["params"])

        # Run the model
        # f_outputs, g_outputs = self(batch)
        g_outputs = self.g_model(batch)
        # sequence = residue_constants.aatype_to_sequence(batch["aatype"][0, ..., -1])
        sequence = residue_constants.aatypeList_to_sequence(batch["aatype"][..., -1].tolist())
        h_outputs = self.forward_joint(batch, sequence)
        batch = tensor_tree_map(lambda t: t[..., -1], batch)

        # batch["use_clamped_fape"] = 0.
        # f_loss, f_loss_breakdown = self.f_loss(
        #     f_outputs, batch, _return_breakdown=True
        # )
        # self._log(f_loss_breakdown, batch, f_outputs, train=False)

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

        self.log('val/inverse_loss', ce, on_step=False, on_epoch=True, prog_bar=False, logger=False, sync_dist=True)
        self.log('val/inverse_PPL', ppl, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)    # show
        self.log('val/inverse_AAR', aars, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)  


        # Compute h loss
        logits_h = h_outputs["sm"]["seqs_logits"][-1]
        masked_pred_h = logits_h.masked_select(batch["seq_mask"].unsqueeze(-1).to(torch.bool)).view(-1, residue_constants.restype_num + 1) # Nl x 21
        ce_h = F.cross_entropy(masked_pred_h, masked_target)
        ppl_h = ce_h.exp()
        logits_h[..., -1] = -9999 # zero out UNK.
        sampled_seqs_h = logits_h.argmax(dim=-1)    # greedy sampling
        masked_sampled_seqs_h = sampled_seqs_h.masked_select(batch["seq_mask"].to(torch.bool)).view(-1) # N x Nl
        aars_h = masked_sampled_seqs_h.eq(masked_target).float().mean()

        # Log it
        self.log('val/reconstruction_loss', ce_h, on_step=False, on_epoch=True, prog_bar=False, logger=False, sync_dist=True)
        self.log('val/reconstruction_PPL', ppl_h, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)    # show
        self.log('val/reconstruction_AAR', aars_h, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)  

        loss = ce
        self.log(
                f"val/loss", 
                loss, 
                on_step=False, on_epoch=True, logger=True, sync_dist=True
            )

        
    def validation_epoch_end(self, _):
        # Restore the model weights to normal
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
            [{"params":self.g_model.parameters()}],
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
        self.g_ema.load_state_dict(checkpoint["g_ema"])

    def on_save_checkpoint(self, checkpoint):
        checkpoint["g_ema"] = self.g_ema.state_dict()


def main(args):
    if(args.seed is not None):
        seed_everything(args.seed) 
    logging.info(f"GPU availability: {torch.cuda.is_available()}")
    logging.info(f"args.is_antibody is {args.is_antibody}")
    config = model_config(
        name=args.config_preset,
        yaml_config_preset=args.yaml_config_preset,
        train=True, 
        low_prec=(args.precision == 16),
    )
    model_module = OpenFoldWrapper(config)

    logging.info(f"args.resume_model_weights_only is {args.resume_model_weights_only}")
    logging.info(f"args.resume_from_ckpt_forward is {args.resume_from_ckpt_forward}")
    logging.info(f"args.resume_from_ckpt_backward is {args.resume_from_ckpt_backward}")
    if(args.resume_model_weights_only):
        assert (args.resume_from_ckpt_forward is not None) or (args.resume_from_ckpt_backward is not None)

        if args.resume_from_ckpt_forward is not None:
            sd = torch.load(args.resume_from_ckpt_forward, map_location=torch.device('cpu'))
            logging.info("printing loaded state dict for model_f")
            stat_dict_f = {k[len("model."):]:v for k,v in sd["state_dict"].items()}
            ema_f = {k:v for k,v in sd["ema"].items()}
            model_module.f_model.load_state_dict(stat_dict_f)
            logging.info("Successfully loaded model_f weights...")

        if args.resume_from_ckpt_backward is not None:
            sd = torch.load(args.resume_from_ckpt_backward, map_location=torch.device('cpu'))
            logging.info("loading state dict for backward model")
            stat_dict_g = {k[len("model."):]:v for k,v in sd["state_dict"].items()}
            stat_dict_m2s = {}
            for k,v in stat_dict_g.items():
                if k in ["evoformer.linear.weight", "evoformer.linear.bias"]:
                    stat_dict_m2s[k[len("evoformer.linear."):]] = v
            model_module.g_model.load_state_dict(stat_dict_g, strict=False)
            model_module.g_model.linear_m2s.load_state_dict(stat_dict_m2s)
            model_module.g_model.eval()
            logging.info("Successfully loaded backward model weights...")

    parallel_data_module = OpenFoldDataModule(
        config=config.data, 
        batch_seed=args.seed,
        **vars(args)
    )

    parallel_data_module.prepare_data()
    parallel_data_module.setup()

    # process fasta file
    # sequence_data_module = OpenFoldDataModule(
    #     config=config.data, 
    #     batch_seed=args.seed,
    #     train_data_dir=args.fasta_dir,
    #     train_epoch_len=args.train_epoch_len,
    #     is_antibody=args.is_antibody,
    # )
    # sequence_data_module.prepare_data()
    # sequence_data_module.setup()

    callbacks = []
    if(args.checkpoint_every_epoch):
        mc = ModelCheckpoint(
            filename="epoch{epoch:02d}-step{step}-val_loss={val/loss:.3f}",
            auto_insert_metric_name=False,
            monitor="val/loss",
            mode="min",
            every_n_epochs=1,
            save_last=False,
            save_top_k=2,
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

    if(args.resume_model_weights_only):
        ckpt_path = None
    else:
        ckpt_path = args.resume_from_ckpt
        print(f">>> full training process is retrieved")

    # multi data module training
    train_dataloader=parallel_data_module.train_dataloader()
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
        "--resume_from_ckpt_forward", type=str, default=None,
        help="Path to a model checkpoint from which to restore model state of folding model"
    )
    parser.add_argument(
        "--resume_from_ckpt_backward", type=str, default=None,
        help="Path to a model checkpoint from which to restore model state of inverse folding model"
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

    # args = parser.parse_args()
    # for k, v in vars(args).items():
    #     logging.info(f"key: {k}")
    #     logging.info(f"value: {v}")
    #     logging.info(f"----------")
    # logging.info(f"args.resume_model_weights_only is {args.resume_model_weights_only}")
    # logging.info(f"args.resume_model_weights_only is {args.resume_model_weights_only}")

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
