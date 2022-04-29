import csv
import copy
from functools import partial
import json
import logging
logging.basicConfig(level=logging.WARNING)
import os
import glob
import pickle
from typing import Optional, Sequence, List, Any

import ml_collections as mlc
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import RandomSampler

from openfold.data import (
    data_pipeline,
    feature_pipeline,
    mmcif_parsing,
    templates,
)
from openfold.utils.tensor_utils import tensor_tree_map, dict_multimap


class OpenFoldSingleDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_dir: str,
        alignment_dir: str,
        embedding_dir: str,
        attn_dir: str,
        pred_pdb_dir: str,
        template_mmcif_dir: str,
        max_template_date: str,
        config: mlc.ConfigDict,
        kalign_binary_path: str = '/usr/bin/kalign',
        max_template_hits: int = 4,
        obsolete_pdbs_file_path: Optional[str] = None,
        template_release_dates_cache_path: Optional[str] = None,
        shuffle_top_k_prefiltered: Optional[int] = None,
        treat_pdb_as_distillation: bool = True,
        mapping_path: Optional[str] = None,
        mode: str = "train", 
        _output_raw: bool = False,
        sabdab_summary_file: Optional[str] = None,
        _alignment_index: Optional[Any] = None,
    ):
        """
            Args:
                data_dir:
                    A path to a directory containing mmCIF files (in train
                    mode) or FASTA files (in inference mode).
                alignment_dir:
                    A path to a directory containing only data in the format 
                    output by an AlignmentRunner 
                    (defined in openfold.features.alignment_runner).
                    I.e. a directory of directories named {PDB_ID}_{CHAIN_ID}
                    or simply {PDB_ID}, each containing .a3m, .sto, and .hhr
                    files.
                embedding_dir:
                    A path to a directory containing pre-trained residue embedding
                    from protein language models, e.g., ESM-1b.
                template_mmcif_dir:
                    Path to a directory containing template mmCIF files.
                config:
                    A dataset config object. See openfold.config
                kalign_binary_path:
                    Path to kalign binary.
                max_template_hits:
                    An upper bound on how many templates are considered. During
                    training, the templates ultimately used are subsampled
                    from this total quantity.
                template_release_dates_cache_path:
                    Path to the output of scripts/generate_mmcif_cache.
                obsolete_pdbs_file_path:
                    Path to the file containing replacements for obsolete PDBs.
                shuffle_top_k_prefiltered:
                    Whether to uniformly shuffle the top k template hits before
                    parsing max_template_hits of them. Can be used to
                    approximate DeepMind's training-time template subsampling
                    scheme much more performantly.
                treat_pdb_as_distillation:
                    Whether to assume that .pdb files in the data_dir are from
                    the self-distillation set (and should be subjected to
                    special distillation set preprocessing steps).
                mode:
                    "train", "val", or "predict"
        """
        super(OpenFoldSingleDataset, self).__init__()
        self.data_dir = data_dir
        self.alignment_dir = alignment_dir
        self.embedding_dir = embedding_dir
        self.attn_dir = attn_dir
        self.pred_pdb_dir = pred_pdb_dir
        self.config = config
        self.treat_pdb_as_distillation = treat_pdb_as_distillation
        self.mode = mode
        self._output_raw = _output_raw
        self._alignment_index = _alignment_index

        valid_modes = ["train", "eval", "predict"]
        if(mode not in valid_modes):
            raise ValueError(f'mode must be one of {valid_modes}')

        if(template_release_dates_cache_path is None):
            logging.warning(
                "Template release dates cache does not exist. Remember to run "
                "scripts/generate_mmcif_cache.py before running OpenFold"
            )

        if(_alignment_index is not None):
            self._chain_ids = list(_alignment_index.keys())
        elif(mapping_path is None):
            #self._chain_ids = list(os.listdir(alignment_dir))
            self._chain_ids = [
                os.path.splitext(name)[0] for name in os.listdir(data_dir)
            ]
        else:
            with open(mapping_path, "r") as f:
                self._chain_ids = [l.strip() for l in f.readlines()]
        
        self._chain_id_to_idx_dict = {
            chain: i for i, chain in enumerate(self._chain_ids)
        }

        # template_featurizer = templates.TemplateHitFeaturizer(
        #     mmcif_dir=template_mmcif_dir,
        #     max_template_date=max_template_date,
        #     max_hits=max_template_hits,
        #     kalign_binary_path=kalign_binary_path,
        #     release_dates_path=template_release_dates_cache_path,
        #     obsolete_pdbs_path=obsolete_pdbs_file_path,
        #     _shuffle_top_k_prefiltered=shuffle_top_k_prefiltered,
        # )
        template_featurizer = None
        logging.warning(
            "'template_featurizer' in OpenFoldSingleDataset is set as None."
        )

        self.data_pipeline = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )

        if(not self._output_raw):
            self.feature_pipeline = feature_pipeline.FeaturePipeline(config)
            
        self.resolution = {}
        if sabdab_summary_file is not None:
            with open(sabdab_summary_file, "r") as fin:
                reader = csv.reader(fin, delimiter="\t")
                fields = next(reader)
                pdb_index = fields.index("pdb")
                res_index = fields.index("resolution")
                for values in reader:
                    if pdb_index < len(values) and res_index < len(values):
                        self.resolution[values[pdb_index]] = float(values[res_index])

    def _parse_mmcif(self, path, file_id, chain_id, alignment_dir, embedding_dir, _alignment_index):
        with open(path, 'r') as f:
            mmcif_string = f.read()

        mmcif_object = mmcif_parsing.parse(
            file_id=file_id, mmcif_string=mmcif_string
        )

        # Crash if an error is encountered. Any parsing errors should have
        # been dealt with at the alignment stage.
        if(mmcif_object.mmcif_object is None):
            raise list(mmcif_object.errors.values())[0]

        mmcif_object = mmcif_object.mmcif_object

        data = self.data_pipeline.process_mmcif(
            mmcif=mmcif_object,
            alignment_dir=alignment_dir,
            chain_id=chain_id,
            embedding_dir=embedding_dir,
            _alignment_index=_alignment_index,
        )

        return data

    def chain_id_to_idx(self, chain_id):
        return self._chain_id_to_idx_dict[chain_id]

    def idx_to_chain_id(self, idx):
        return self._chain_ids[idx]

    def __getitem__(self, idx):
        name = self.idx_to_chain_id(idx)
        alignment_dir = os.path.join(self.alignment_dir, name)

        if not os.path.exists(alignment_dir):
            # disable alignment
            alignment_dir = self.alignment_dir

        if self.embedding_dir is not None:
            embedding_dir = os.path.join(self.embedding_dir, name)
            if not os.path.exists(embedding_dir):
                raise ValueError(
                    f"""
                     The embedding dir {embedding_dir} does not exist.
                     """
                )
        else:
            embedding_dir = None
        
        if self.attn_dir is not None:
            attn_path_H = os.path.join(self.attn_dir, name + '_H.oaspt')
            attn_path_L = os.path.join(self.attn_dir, name + '_L.oaspt')
        else:
            attn_path_H = attn_path_L = None
            
        _alignment_index = None
        if(self._alignment_index is not None):
            alignment_dir = self.alignment_dir
            _alignment_index = self._alignment_index[name]

        if(self.mode == 'train' or self.mode == 'eval'):
            spl = name.rsplit('_', 1)
            if(len(spl) == 2):
                file_id, chain_id = spl
            else:
                file_id, = spl
                chain_id = None

            path = os.path.join(self.data_dir, file_id)
            if(os.path.exists(path + ".cif")):
                data = self._parse_mmcif(
                    path + ".cif", file_id, chain_id, alignment_dir, embedding_dir, _alignment_index,
                )
            elif(os.path.exists(path + ".core")):
                data = self.data_pipeline.process_core(
                    path + ".core", alignment_dir, embedding_dir, _alignment_index,
                )
            elif(os.path.exists(path + ".pdb")):
                resolution = self.resolution.get(file_id, 0)
                data = self.data_pipeline.process_pdb(
                    pdb_path=path + ".pdb",
                    alignment_dir=alignment_dir,
                    is_distillation=self.treat_pdb_as_distillation,
                    chain_id=chain_id,
                    embedding_dir=embedding_dir,
                    attn_path_H=attn_path_H,
                    attn_path_L=attn_path_L,
                    resolution=resolution,
                    _alignment_index=_alignment_index,
                )
            else:
                raise ValueError("Invalid file type")

            if self.pred_pdb_dir is not None:
                pred_pdb_path = glob.glob(os.path.join(self.pred_pdb_dir, f"{file_id}_*.pdb"))
                assert len(pred_pdb_path) == 1, f"{len(pred_pdb_path)} predictions are found!"
                pred_pdb_path = pred_pdb_path[0]

                pred_data = self.data_pipeline.process_pdb(
                    pdb_path=pred_pdb_path,
                    alignment_dir=alignment_dir,
                    is_distillation=self.treat_pdb_as_distillation,
                    chain_id=chain_id,
                    embedding_dir=embedding_dir,
                    attn_path_H=attn_path_H,
                    attn_path_L=attn_path_L,
                    _alignment_index=_alignment_index,
                )
                data["pred_atom_positions"] = pred_data["all_atom_positions"]
        else:
            path = os.path.join(name, name + ".fasta")
            data = self.data_pipeline.process_fasta(
                fasta_path=path,
                alignment_dir=alignment_dir,
                embedding_dir=embedding_dir,
                attn_path_H=attn_path_H,
                attn_path_L=attn_path_L,
                _alignment_index=_alignment_index,
            )

        if(self._output_raw):
            return data

        feats = self.feature_pipeline.process_features(
            data, self.mode 
        )

        return feats

    def __len__(self):
        return len(self._chain_ids) 


def deterministic_train_filter(
    chain_data_cache_entry: Any,
    max_resolution: float = 9.,
    max_single_aa_prop: float = 0.8,
) -> bool:
    """Training data filters as described in Supplement. 1.2.5"""
    # Hard filters. Input mmCIFs are restricted to have resolution less than 9A.
    resolution = chain_data_cache_entry.get("resolution", None)
    if(resolution is not None and resolution > max_resolution):
        return False

    seq = chain_data_cache_entry["seq"]
    counts = {}
    for aa in seq:
        counts.setdefault(aa, 0)
        counts[aa] += 1
    largest_aa_count = max(counts.values())
    largest_single_aa_prop = largest_aa_count / len(seq)
    
    # Sequences are filtered out when any single amino acid
    # accounts for more than 80% of the input primary sequence.
    if(largest_single_aa_prop > max_single_aa_prop):
        return False

    return True


def get_stochastic_train_filter_prob(
    chain_data_cache_entry: Any,
) -> List[float]:
    """Training data filters as described in Supplement. 1.2.5"""
    # Stochastic filters
    probabilities = []
    
    cluster_size = chain_data_cache_entry.get("cluster_size", None)
    if(cluster_size is not None and cluster_size > 0):
        probabilities.append(1 / cluster_size)
    
    chain_length = len(chain_data_cache_entry["seq"])
    probabilities.append((1 / 512) * (max(min(chain_length, 512), 256)))

    # Risk of underflow here?
    out = 1
    for p in probabilities:
        out *= p

    return out


class OpenFoldDataset(torch.utils.data.Dataset):
    """
        Implements the stochastic filters applied during AlphaFold's training.
        Because samples are selected from constituent datasets randomly, the
        length of an OpenFoldFilteredDataset is arbitrary. Samples are selected
        and filtered once at initialization.
    """
    def __init__(self,
        datasets: Sequence[OpenFoldSingleDataset],
        probabilities: Sequence[int],
        epoch_len: int,
        chain_data_cache_paths: List[Optional[str]],
        generator: torch.Generator = None,
        _roll_at_init: bool = True,
    ):
        self.datasets = datasets
        self.probabilities = probabilities
        self.epoch_len = epoch_len
        self.generator = generator
        
        if None in chain_data_cache_paths:
            self.chain_data_caches = None
        else:
            self.chain_data_caches = []
            for path in chain_data_cache_paths:
                with open(path, "r") as fp:
                    self.chain_data_caches.append(json.load(fp))

        def looped_shuffled_dataset_idx(dataset_len):
            while True:
                # Uniformly shuffle each dataset's indices
                weights = [1. for _ in range(dataset_len)]
                shuf = torch.multinomial(
                    torch.tensor(weights),
                    num_samples=dataset_len,
                    replacement=False,
                    generator=self.generator,
                )
                for idx in shuf:
                    yield idx

        def looped_samples(dataset_idx):
            max_cache_len = int(epoch_len * probabilities[dataset_idx])
            dataset = self.datasets[dataset_idx]
            idx_iter = looped_shuffled_dataset_idx(len(dataset))
            
            # if chain_data_cache_paths is not provided,
            # we ignore all train_filters described in Supplement. 1.2.5
            chain_data_cache = self.chain_data_caches[dataset_idx] \
                if self.chain_data_caches is not None else None
            
            while True:
                weights = []
                idx = []
                for _ in range(max_cache_len):
                    candidate_idx = next(idx_iter)
                    chain_id = dataset.idx_to_chain_id(candidate_idx)
                    
                    p = 1.0 # if not filtered, always accept the candidate
                    if chain_data_cache is not None: # perform filtering
                        chain_data_cache_entry = chain_data_cache[chain_id]
                        if(not deterministic_train_filter(chain_data_cache_entry)):
                            continue

                        p = get_stochastic_train_filter_prob(
                            chain_data_cache_entry,
                        )
                    weights.append([1. - p, p])
                    idx.append(candidate_idx)

                samples = torch.multinomial(
                    torch.tensor(weights),
                    num_samples=1,
                    generator=self.generator,
                ) # [max_cache_len, 2]
                # [max_cache_len, ], flags for acceptance
                samples = samples.squeeze(-1)

                cache = [i for i, s in zip(idx, samples) if s]

                for datapoint_idx in cache:
                    yield datapoint_idx

        self._samples = [looped_samples(i) for i in range(len(self.datasets))]

        if(_roll_at_init):
            self.reroll()

    def __getitem__(self, idx):
        dataset_idx, datapoint_idx = self.datapoints[idx]
        return self.datasets[dataset_idx][datapoint_idx]

    def __len__(self):
        return self.epoch_len

    def reroll(self):
        dataset_choices = torch.multinomial(
            torch.tensor(self.probabilities),
            num_samples=self.epoch_len,
            replacement=True,
            generator=self.generator,
        )

        self.datapoints = []
        for dataset_idx in dataset_choices:
            samples = self._samples[dataset_idx]
            datapoint_idx = next(samples)
            self.datapoints.append((dataset_idx, datapoint_idx))


class OpenFoldBatchCollator:
    def __init__(self, config, stage="train"):
        self.stage = stage
        self.feature_pipeline = feature_pipeline.FeaturePipeline(config)

    def __call__(self, raw_prots):
        processed_prots = []
        for prot in raw_prots:
            features = self.feature_pipeline.process_features(
                prot, self.stage
            )
            processed_prots.append(features)

        stack_fn = partial(torch.stack, dim=0)
        return dict_multimap(stack_fn, processed_prots) 


class OpenFoldDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, config, stage="train", generator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.stage = stage    

        if(generator is None):
            generator = torch.Generator()
        
        self.generator = generator
        self._prep_batch_properties_probs()

    def _prep_batch_properties_probs(self):
        keyed_probs = []
        stage_cfg = self.config[self.stage]

        max_iters = self.config.common.max_recycling_iters
        if(stage_cfg.supervised):
            # Supplement 1.11.5
            # In 90% of training mini-batches the FAPE backbone loss is clamped by e_max = 10A.
            # In the remaining 10% it is not clamped, e_max = + \infty.
            # For side-chains it is always clamped by e_max = 10A.
            clamp_prob = self.config.supervised.clamp_prob
            keyed_probs.append(
                ("use_clamped_fape", [1 - clamp_prob, clamp_prob])
            )
        
        if(stage_cfg.uniform_recycling):
            recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
        else:
            recycling_probs = [
                0. for _ in range(max_iters + 1)
            ]
            recycling_probs[-1] = 1.
        
        keyed_probs.append(
            ("no_recycling_iters", recycling_probs)
        )

        keys, probs = zip(*keyed_probs)
        max_len = max([len(p) for p in probs])
        padding = [[0.] * (max_len - len(p)) for p in probs] 
        
        self.prop_keys = keys
        self.prop_probs_tensor = torch.tensor(
            [p + pad for p, pad in zip(probs, padding)],
            dtype=torch.float32,
        )

    def _add_batch_properties(self, batch):
        samples = torch.multinomial(
            self.prop_probs_tensor,
            num_samples=1, # 1 per row
            replacement=True,
            generator=self.generator
        )

        aatype = batch["aatype"]
        batch_dims = aatype.shape[:-2]
        recycling_dim = aatype.shape[-1]
        no_recycling = recycling_dim
        for i, key in enumerate(self.prop_keys):
            sample = int(samples[i][0])
            sample_tensor = torch.tensor(
                sample, 
                device=aatype.device, 
                requires_grad=False
            )
            orig_shape = sample_tensor.shape
            sample_tensor = sample_tensor.view(
                (1,) * len(batch_dims) + sample_tensor.shape + (1,)
            )
            sample_tensor = sample_tensor.expand(
                batch_dims + orig_shape + (recycling_dim,)
            )
            batch[key] = sample_tensor

            if(key == "no_recycling_iters"):
                no_recycling = sample 
        
        resample_recycling = lambda t: t[..., :no_recycling + 1]
        batch = tensor_tree_map(resample_recycling, batch)

        return batch

    def __iter__(self):
        it = super().__iter__()

        def _batch_prop_gen(iterator):
            for batch in iterator:
                yield self._add_batch_properties(batch)

        return _batch_prop_gen(it)


class OpenFoldDataModule(pl.LightningDataModule):
    def __init__(self,
        config: mlc.ConfigDict,
        template_mmcif_dir: str,
        max_template_date: str,
        train_data_dir: Optional[str] = None,
        train_alignment_dir: Optional[str] = None,
        train_embedding_dir: Optional[str] = None,
        train_attn_dir: Optional[str] = None,
        train_chain_data_cache_path: Optional[str] = None,
        pred_train_pdb_dir: Optional[str] = None,
        distillation_data_dir: Optional[str] = None,
        distillation_alignment_dir: Optional[str] = None,
        distillation_embedding_dir: Optional[str] = None,
        distillation_attn_dir: Optional[str] = None,
        distillation_chain_data_cache_path: Optional[str] = None,
        val_data_dir: Optional[str] = None,
        val_alignment_dir: Optional[str] = None,
        val_embedding_dir: Optional[str] = None,
        val_attn_dir: Optional[str] = None,
        pred_val_pdb_dir: Optional[str] = None,
        predict_data_dir: Optional[str] = None,
        predict_alignment_dir: Optional[str] = None,
        predict_embedding_dir: Optional[str] = None,
        predict_attn_dir: Optional[str] = None,
        kalign_binary_path: str = '/usr/bin/kalign',
        train_mapping_path: Optional[str] = None,
        distillation_mapping_path: Optional[str] = None,
        obsolete_pdbs_file_path: Optional[str] = None,
        template_release_dates_cache_path: Optional[str] = None,
        batch_seed: Optional[int] = None,
        train_epoch_len: Optional[int] = None,
        sabdab_summary_file: Optional[str] = None,
        _alignment_index_path: Optional[str] = None,
        **kwargs
    ):
        super(OpenFoldDataModule, self).__init__()

        self.config = config
        self.template_mmcif_dir = template_mmcif_dir
        self.max_template_date = max_template_date
        self.train_data_dir = train_data_dir
        self.train_alignment_dir = train_alignment_dir
        self.train_embedding_dir = train_embedding_dir
        self.train_attn_dir = train_attn_dir
        self.train_chain_data_cache_path = train_chain_data_cache_path
        self.pred_train_pdb_dir = pred_train_pdb_dir
        self.distillation_data_dir = distillation_data_dir
        self.distillation_alignment_dir = distillation_alignment_dir
        self.distillation_embedding_dir = distillation_embedding_dir
        self.distillation_attn_dir = distillation_attn_dir
        self.distillation_chain_data_cache_path = (
            distillation_chain_data_cache_path
        )
        self.val_data_dir = val_data_dir
        self.val_alignment_dir = val_alignment_dir
        self.val_embedding_dir = val_embedding_dir
        self.val_attn_dir = val_attn_dir
        self.pred_val_pdb_dir = pred_val_pdb_dir
        self.predict_data_dir = predict_data_dir
        self.predict_alignment_dir = predict_alignment_dir
        self.predict_embedding_dir = predict_embedding_dir
        self.predict_attn_dir = predict_attn_dir
        self.kalign_binary_path = kalign_binary_path
        self.train_mapping_path = train_mapping_path
        self.distillation_mapping_path = distillation_mapping_path
        self.template_release_dates_cache_path = (
            template_release_dates_cache_path
        )
        self.obsolete_pdbs_file_path = obsolete_pdbs_file_path
        self.batch_seed = batch_seed
        self.train_epoch_len = train_epoch_len
        self.sabdab_summary_file = sabdab_summary_file

        if(self.train_data_dir is None and self.predict_data_dir is None):
            raise ValueError(
                'At least one of train_data_dir or predict_data_dir must be '
                'specified'
            )

        self.training_mode = self.train_data_dir is not None

        if(self.training_mode and train_alignment_dir is None):
            raise ValueError(
                'In training mode, train_alignment_dir must be specified'
            )
        elif(not self.training_mode and predict_alignment_dir is None):
            raise ValueError(
                'In inference mode, predict_alignment_dir must be specified'
            )      
        elif(val_data_dir is not None and val_alignment_dir is None):
            raise ValueError(
                'If val_data_dir is specified, val_alignment_dir must '
                'be specified as well'
        )

        # An ad-hoc measure for our particular filesystem restrictions
        self._alignment_index = None
        if(_alignment_index_path is not None):
            with open(_alignment_index_path, "r") as fp:
                self._alignment_index = json.load(fp)

    def setup(self):
        # Most of the arguments are the same for the three datasets 
        dataset_gen = partial(OpenFoldSingleDataset,
            template_mmcif_dir=self.template_mmcif_dir,
            max_template_date=self.max_template_date,
            config=self.config,
            kalign_binary_path=self.kalign_binary_path,
            template_release_dates_cache_path=
                self.template_release_dates_cache_path,
            obsolete_pdbs_file_path=
                self.obsolete_pdbs_file_path,
        )

        if(self.training_mode):
            train_dataset = dataset_gen(
                data_dir=self.train_data_dir,
                alignment_dir=self.train_alignment_dir,
                embedding_dir=self.train_embedding_dir,
                attn_dir=self.train_attn_dir,
                pred_pdb_dir=self.pred_train_pdb_dir,
                mapping_path=self.train_mapping_path,
                max_template_hits=self.config.train.max_template_hits,
                shuffle_top_k_prefiltered=
                    self.config.train.shuffle_top_k_prefiltered,
                treat_pdb_as_distillation=False,
                mode="train",
                _output_raw=True,
                sabdab_summary_file=self.sabdab_summary_file,
                _alignment_index=self._alignment_index,
            )

            distillation_dataset = None
            if(self.distillation_data_dir is not None):
                distillation_dataset = dataset_gen(
                    data_dir=self.distillation_data_dir,
                    alignment_dir=self.distillation_alignment_dir,
                    embedding_dir=self.distillation_embedding_dir,
                    attn_dir=self.distillation_attn_dir,
                    mapping_path=self.distillation_mapping_path,
                    max_template_hits=self.train.max_template_hits,
                    treat_pdb_as_distillation=True,
                    mode="train",
                    _output_raw=True,
                    sabdab_summary_file=self.sabdab_summary_file,
                )

                d_prob = self.config.train.distillation_prob
           
            if(distillation_dataset is not None):
                datasets = [train_dataset, distillation_dataset]
                d_prob = self.config.train.distillation_prob
                probabilities = [1 - d_prob, d_prob]
                chain_data_cache_paths = [
                    self.train_chain_data_cache_path,
                    self.distillation_chain_data_cache_path,
                ]
            else:
                datasets = [train_dataset]
                probabilities = [1.]   
                chain_data_cache_paths = [
                    self.train_chain_data_cache_path,
                ]
            train_epoch_len = self.train_epoch_len or sum([len(_) for _ in datasets])
            self.train_dataset = OpenFoldDataset(
                datasets=datasets,
                probabilities=probabilities,
                epoch_len=train_epoch_len,
                chain_data_cache_paths=chain_data_cache_paths,
                _roll_at_init=False,
            )
    
            if(self.val_data_dir is not None):
                self.eval_dataset = dataset_gen(
                    data_dir=self.val_data_dir,
                    alignment_dir=self.val_alignment_dir,
                    embedding_dir=self.val_embedding_dir,
                    attn_dir=self.val_attn_dir,
                    pred_pdb_dir=self.pred_val_pdb_dir,
                    mapping_path=None,
                    max_template_hits=self.config.eval.max_template_hits,
                    mode="eval",
                    _output_raw=True,
                    sabdab_summary_file=self.sabdab_summary_file,
                )
            else:
                self.eval_dataset = None
        else:           
            self.predict_dataset = dataset_gen(
                data_dir=self.predict_data_dir,
                alignment_dir=self.predict_alignment_dir,
                embedding_dir=self.predict_embedding_dir,
                attn_dir=self.predict_attn_dir,
                mapping_path=None,
                max_template_hits=self.config.predict.max_template_hits,
                mode="predict",
            )

    def _gen_dataloader(self, stage):
        generator = torch.Generator()
        if(self.batch_seed is not None):
            generator = generator.manual_seed(self.batch_seed)

        dataset = None
        if(stage == "train"):
            dataset = self.train_dataset
            # Filter the dataset, if necessary
            dataset.reroll()
        elif(stage == "eval"):
            dataset = self.eval_dataset
        elif(stage == "predict"):
            dataset = self.predict_dataset
        else:
            raise ValueError("Invalid stage")

        batch_collator = OpenFoldBatchCollator(self.config, stage)

        dl = OpenFoldDataLoader(
            dataset,
            config=self.config,
            stage=stage,
            generator=generator,
            batch_size=self.config.data_module.data_loaders.batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            collate_fn=batch_collator,
        )

        return dl

    def train_dataloader(self):
        return self._gen_dataloader("train") 

    def val_dataloader(self):
        if(self.eval_dataset is not None):
            return self._gen_dataloader("eval")
        return None

    def predict_dataloader(self):
        return self._gen_dataloader("predict") 


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, batch_path):
        with open(batch_path, "rb") as f:
            self.batch = pickle.load(f)

    def __getitem__(self, idx):
        return copy.deepcopy(self.batch)

    def __len__(self):
        return 1000


class DummyDataLoader(pl.LightningDataModule):
    def __init__(self, batch_path):
        super().__init__()
        self.dataset = DummyDataset(batch_path)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset)
