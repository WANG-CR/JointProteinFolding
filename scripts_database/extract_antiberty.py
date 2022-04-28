import transformers
from torch.nn import functional as F
from torch import nn
import torch
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import os
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)


class AntiBERTy(nn.Module):
    def __init__(self, config, tokenizer):
        super(AntiBERTy, self).__init__()
        self.bert_model = transformers.BertModel(config)
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer.vocab)
        bert_layers = self.bert_model.config.num_hidden_layers
        self.bert_feat_dim = self.bert_model.config.hidden_size
        self.bert_attn_dim = bert_layers * self.bert_model.config.num_attention_heads

    def get_tokens(
        self,
        seq,
    ):
        assert isinstance(seq, list) and isinstance(seq[0], str)
        seqs = [" ".join(list(s)) for s in seq]
        output = self.tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
        )
        device = self.bert_model.embeddings.word_embeddings.weight.device
        tokens = output["input_ids"].to(device)
        masks = output["attention_mask"].to(device)

        return tokens, masks

    def forward(self, seq):
        tokens, masks = self.get_tokens(seq)  # [*, seq_len]
        bert_output = self.bert_model(
            tokens,
            attention_mask=masks,
            output_hidden_states=True,
            output_attentions=True,
        )

        feats = bert_output.hidden_states[-1]  # [*, seq_len, n_hidden]
        # [*, n_head * n_layer, seq_len, seq_len]
        attns = torch.cat(
            bert_output.attentions,
            dim=1,
        )
        attns = torch.permute(attns, (0, 2, 3, 1))

        return feats, attns, masks


def parse_fasta(fasta_string: str) -> Tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
        fasta_string: The string contents of a FASTA file.

    Returns:
        A tuple of two lists:
        * A list of sequences.
        * A list of sequence descriptions taken from the comment lines. In the
            same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith(">"):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append("")
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


def main(args):

    # 1. get fastas and prepare data
    with open(args.fasta_path, "r") as fp:
        fasta_str = fp.read()
    input_seqs, input_descs = parse_fasta(fasta_str)
    batch_size = 100
    num_batches = (len(input_seqs) + batch_size - 1) // batch_size

    # 2. init model. from ckpt provided by IgFold.
    ckpt = torch.load(args.ckpt_path, map_location='cpu')
    sd = ckpt["state_dict"]
    sd = {
        k: v for k, v in sd.items() if k[:len("bert_model")] == "bert_model"
    }
    config = ckpt["hyper_parameters"]["config"]
    tokenizer = config["tokenizer"]
    model = AntiBERTy(config["bert_config"], tokenizer)
    model.load_state_dict(sd)

    if args.gpu:
        model = model.cuda(0)
    model.eval()

    logging.info(
        f"saving node_embs embeddings into {args.output_node_dir}..."
    )
    if not os.path.exists(args.output_node_dir):
        os.makedirs(args.output_node_dir, exist_ok=True)
    logging.info(
        f"saving edge_embs embeddings into {args.output_edge_dir}..."
    )
    if not os.path.exists(args.output_edge_dir):
        os.makedirs(args.output_edge_dir, exist_ok=True)

    for batch_id in range(num_batches):
        logging.info(f"processing {batch_id}-th batch...")
        batch_seqs = input_seqs[batch_id *
                                batch_size: (batch_id+1) * batch_size]
        batch_descs = input_descs[batch_id *
                                  batch_size: (batch_id+1) * batch_size]
        node_embs = []
        edge_embs = []

        with torch.no_grad():
            feats, attns, masks = model(batch_seqs)
            feats = feats.cpu()
            attns = attns.cpu()
            masks = masks.cpu()

        for feat, attn, mask in zip(feats, attns, masks):
            # feat: [max_len, n_hidden]
            # attn: [max_len, max_len, n_head * n_layer]
            # mask: [max_len]
            feat = feat[mask == 1]  # [seq_len, n_hidden]
            # [seq_len, seq_len, n_head * n_layer]
            attn = attn[mask == 1, :][:, mask == 1]

            feat = feat[1:-1, :]
            attn = attn[1:-1, 1:-1, :]

            node_embs.append(feat)
            edge_embs.append(attn)

        for tag, emb in zip(batch_descs, node_embs):
            output_path = os.path.join(args.output_node_dir, tag + '.oaspt')
            torch.save(emb.clone(), output_path)

        for tag, emb in zip(batch_descs, edge_embs):
            output_path = os.path.join(args.output_edge_dir, tag + '.oaspt')
            torch.save(emb.clone(), output_path)


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
        "fasta_path", type=str,
        help="Path to the fasta file"
    )
    parser.add_argument(
        "output_node_dir", type=str,
        help="Path to a output directory containing merged esm representations"
    )
    parser.add_argument(
        "output_edge_dir", type=str,
        help="Path to a output directory containing merged esm representations"
    )
    parser.add_argument(
        "ckpt_path", type=str,
        help="Path to a ckpt"
    )
    parser.add_argument(
        "--gpu", type=bool_type, default=True,
        help="""Whether to use gpu"""
    )

    args = parser.parse_args()

    main(args)
