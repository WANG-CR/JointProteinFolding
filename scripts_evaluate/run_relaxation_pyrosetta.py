import pyrosetta
import os
import argparse
from datetime import date
import logging
logging.basicConfig(level=logging.INFO)

init_string = "-mute all -detect_disulf true -detect_disulf_tolerance 1.5 -check_cdr_chainbreaks false"
pyrosetta.init(init_string)


def get_min_mover(
        max_iter: int = 1000) -> pyrosetta.rosetta.protocols.moves.Mover:
    """
    Create full-atom minimization mover
    """

    sf = pyrosetta.create_score_function('ref2015_cst')
    sf.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded,
        1,
    )
    sf.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.pro_close,
        0,
    )
    sf.set_weight(
        pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint,
        1,
    )

    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(False)
    mmap.set_jump(True)
    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap,
        sf,
        'lbfgs_armijo_nonmonotone',
        0.0001,
        True,
    )
    min_mover.max_iter(max_iter)
    min_mover.cartesian(True)

    return min_mover


def relax_pdb(old_pdb, relaxed_pdb):
    pose = pyrosetta.pose_from_pdb(old_pdb)

    cst_mover = pyrosetta.rosetta.protocols.relax.AtomCoordinateCstMover()
    cst_mover.cst_sidechain(False)
    cst_mover.apply(pose)

    min_mover = get_min_mover(50)
    min_mover.apply(pose)
    idealize_mover = pyrosetta.rosetta.protocols.idealize.IdealizeMover()
    idealize_mover.apply(pose)

    pose.dump_pdb(relaxed_pdb)


def main(args):
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir, exist_ok=True)
    logging.info(f"target dir: {args.target_dir}")
    logging.info("start relaxation using pyrosetta")

    jobs = [x for x in os.listdir(args.source_dir) if x.endswith('.pdb')]
    for fname in jobs:
        pdbid = fname[:4]
        # if pdbid in test_list:
        logging.info(f"relaxing {pdbid}...")
        src_path = os.path.join(args.source_dir, fname)
        tgt_path = os.path.join(args.target_dir, fname)
        relax_pdb(src_path, tgt_path)


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
        "source_dir", type=str,
        help="Path to the pred_pdb"
    )
    parser.add_argument(
        "target_dir", type=str,
        help="Path to the native_pdb"
    )
    args = parser.parse_args()

    main(args)