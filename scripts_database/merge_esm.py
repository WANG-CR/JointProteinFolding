import argparse
import os
import logging
logging.basicConfig(level=logging.INFO)
import shutil
import torch


def merge_esm_embedding(pdbid, args):
    fpath_H = os.path.join(args.data_dir, pdbid + '_H.pt')
    fpath_L = os.path.join(args.data_dir, pdbid + '_L.pt')
    embed_H = torch.load(fpath_H)['representations'][33]
    embed_L = torch.load(fpath_L)['representations'][33]
    embed_HL = torch.cat([embed_H, embed_L], dim=0) # (seq_H + seq_L, 1280)
    return embed_HL    


def merge_oas_unpaired_embedding(pdbid, args):
    fpath_H = os.path.join(args.data_dir, pdbid + '_H.oaspt')
    fpath_L = os.path.join(args.data_dir, pdbid + '_L.oaspt')
    embed_H = torch.load(fpath_H)
    embed_L = torch.load(fpath_L)
    embed_HL = torch.cat([embed_H, embed_L], dim=0) # (seq_H + seq_L, 768)
    return embed_HL
          
          
def main(args):
    job_pools = set()
    job_candidate = [x for x in os.listdir(args.data_dir) if x.endswith('.pt')]
    for job in job_candidate:
        job_pools.add(os.path.splitext(job)[0][:4])
    job_pools = list(job_pools)
    print(f'Got {len(job_pools)} jobs to process.')
    
    fn_dict = {
        "esm1b": merge_esm_embedding,
        "oas_unpaired": merge_oas_unpaired_embedding
    }
    for job in job_pools:
        embed_HL = fn_dict[args.model_name](job, args)
        output_dir = os.path.join(args.output_dir, job)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)        
        torch.save(embed_HL, os.path.join(output_dir, f"{args.model_name}.pt"))
    
    # delete the directory containing unmerged embeddings
    shutil.rmtree(args.data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=str,
        help="Path to a directory containing .pt files"
    )
    parser.add_argument(
        "output_dir", type=str,
        help="Path to a output directory containing merged esm representations"
    )
    
    parser.add_argument(
        "--model_name", type=str, default="esm1b",
        help="name of the language model"
    )    

    args = parser.parse_args()

    main(args)
