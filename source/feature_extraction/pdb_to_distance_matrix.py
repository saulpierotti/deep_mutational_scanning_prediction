#!/usr/bin/python3

"""
Author: Saul Pierotti
Mail: saulpierotti.bioinfo@gmail.com
Last updated: 19/03/2021

This script takes as input a mmcif file and a chain id.
It computes the pairwise distance matrix of the alpha carbons and
saves it in a xz compressed joblib dump.
It can optionally also take a list of pdb ids and do the same
action for all the files with such names in the current directory.
In case of files with more than 1 model, only the model with id 0 (the first
one) is considered.
"""

import argparse
import os

import joblib
import numpy as np
from Bio import PDB, SeqUtils
from scipy import spatial


def get_distance_matrix(mmcif_file, chain_id):
    """
    Given a protein structure in mmcif format and a chain id, extract the
    residue type, the coordinate of each residue, and the residue numbering.
    Compute all the pairwise euclidean distances among residues. Returns a
    dictionary containing all of these data.
    """

    parser = PDB.MMCIFParser()
    structure = parser.get_structure("_", mmcif_file)
    out = {
        # this is the residue identity (which aminoacid it is)
        "residue": [],
        # the xyz coordinates for the beta carbon (alpha for GLY)
        "coordinates": [],
        # this corresponds to the numerical part of PDB_BEG and PDB_END in the
        # sifts mapping table (es. 1 for residue 1A)
        "resseq": [],
        # this corresponds to the letteral part of PDB_BEG and PDB_END in the
        # sifts mapping table (es. A for residue 1A)
        "icode": [],
    }

    matching_chains = 0

    for model in structure:
        if model.id == 0:
            print("Processing model:", model.id)

            for chain in model:
                if chain.id != chain_id:
                    continue
                matching_chains += 1

                for residue in chain.get_residues():
                    het_field = residue.id[0]

                    # discard HETATM records

                    if het_field != " ":
                        continue
                    out["residue"].append(residue.resname)

                    if residue.resname == "GLY":
                        # GLY does not have a beta carbon
                        out["coordinates"].append(residue["CA"].get_coord())
                    else:
                        out["coordinates"].append(residue["CB"].get_coord())
                    out["resseq"].append(residue.id[1])
                    out["icode"].append(residue.id[2])
        else:
            print("Skipping model:", model.id)
    assert matching_chains == 1

    out["coordinates"] = np.array(out["coordinates"], dtype=float)
    out["resseq"] = np.array(out["resseq"], dtype=int)
    out["icode"] = np.array(out["icode"], dtype=str)
    # NaN is not defined in a str array, and I need to represent in a way which
    # is coherent with the way pandas represents it
    out["icode"] = np.where(out["icode"] == " ", "nan", out["icode"])
    # the Minkowski 2-norm is the euclidean distance
    out["distance_matrix"] = spatial.distance_matrix(
        out["coordinates"], out["coordinates"], p=2
    )
    out["sequence"] = "".join(
        [
            SeqUtils.IUPACData.protein_letters_3to1[r.capitalize()]
            for r in out["residue"]
        ]
    )
    out["pdb_id"] = mmcif_file.split(".")[0]
    out["chain_id"] = chain_id

    return out


def main(args):
    """
    Main function
    """
    assert os.path.isfile(args.i)
    assert args.i.endswith(".cif")
    assert len(args.c) == 1
    assert args.o.endswith(".pdb_distance_matrix.joblib.xz")
    assert not os.path.isfile(args.o)
    out_dict = get_distance_matrix(args.i, args.c)
    joblib.dump(out_dict, args.o)


def parse_arguments():
    """
    Parse command line arguments.
    """
    description = " ".join(__doc__.splitlines()[4:])
    epilog = ", ".join(__doc__.splitlines()[1:4])
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument(
        "-i",
        type=str,
        help="the mmcif file to be used",
        metavar="<file>",
        required=True,
    )
    parser.add_argument(
        "-o",
        type=str,
        help="the file where to save the joblib dump of the distance matrix",
        metavar="<file>",
        required=True,
    )
    parser.add_argument(
        "-c",
        type=str,
        help="the pdb chain id to be considered",
        metavar="<letter>",
        required=True,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_arguments()
    main(ARGS)
