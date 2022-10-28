import os
import argparse
from tqdm import tqdm

def main(args):
    fasta_path = args.fasta_path
    name = False
    with open(fasta_path, 'r') as file:
        lines = file.readlines()

    chain_ids: list[str] = []
    aastr: list[str] = []

    for line in lines:
        if len(line) == 0:
            continue
        if line.startswith(">") or line.startswith(":"):
            name = True
            chain_ids.append(line[:].strip("\n"))
            # chain_ids.append(line[10:16])
        else:
            if name:
                aastr.append(line.strip("\n").upper())
                name = False
            else:
                aastr[-1] = aastr[-1] + line.strip("\n").upper()
    combined = sorted(
        list(zip(chain_ids, aastr)), key=lambda x: len(x[1])
    )

    output_dir = args.output_dir

    for items in tqdm(combined):
        description = items[0]
        name = items[0][10:16].lower()
        aastr = items[1]
        file_path = os.path.join(output_dir, name + ".fasta")
        with open(file_path, "w+") as f:
            f.write(description+"\n")
            f.write(aastr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta_path", type=str, default=None,
        help="Directory containing training mmCIF files"
    )
    parser.add_argument(
        "--output_dir", type=str, default='invfold_outputs',
        help=(
            "Directory in which to output checkpoints, logs, etc. Ignored "
            "if not on rank 0"
        )
    )
   
    args = parser.parse_args()

    main(args)
