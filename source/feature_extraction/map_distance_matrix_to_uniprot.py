#!/usr/bin/python3

"""
Author: Saul Pierotti
Mail: saulpierotti.bioinfo@gmail.com
Last updated: 22/03/2021

Given the output of the script ./pdb_to_distance_matrix.py, and a sifts mapping,
produces a mapped version of the distance matrix and saves it in a joblib dump.
The dump contains a dictionary with the distance matrix itself and the uniprot
sequence. The uniprot sequence is downloaded dinamycally as requested. The name
of the file which is created is
<uniprot_id>_mapped_<pdb_id>_<chain>.uniprot_distance_matrix.joblib.xz
"""

import argparse
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import requests


def map_d_mat(d_mat_pdb_dict, uniprot_seq, sifts_map):
    """"""

    d_mat_pdb = d_mat_pdb_dict["distance_matrix"]
    pdb_seq = d_mat_pdb_dict["sequence"]
    d_mat_unp = np.empty((len(uniprot_seq), len(uniprot_seq)))
    d_mat_unp[:, :] = np.nan
    # the distance matrices must be square
    assert d_mat_pdb.shape[0] == d_mat_pdb.shape[1]
    assert d_mat_unp.shape[0] == d_mat_unp.shape[1]
    assert d_mat_pdb.shape[0] == len(pdb_seq)

    # get pdb indexes from the resseq and icode values
    pdb_residue_index_map = {
        (str(resseq) + str(icode)): i
        for i, (resseq, icode) in enumerate(
            zip(d_mat_pdb_dict["resseq"], d_mat_pdb_dict["icode"])
        )
    }
    assert len(pdb_residue_index_map) == d_mat_pdb.shape[0]
    unp_b = sifts_map["SP_BEG"] - 1
    unp_e = sifts_map["SP_END"]
    index_shape = unp_b.shape
    assert unp_e.shape == index_shape
    assert unp_b.shape == index_shape
    assert sifts_map["PDB_BEG_NUM"].shape == index_shape
    assert sifts_map["PDB_END_NUM"].shape == index_shape
    assert sifts_map["PDB_BEG_ICODE"].shape == index_shape
    assert sifts_map["PDB_END_ICODE"].shape == index_shape
    index_len = index_shape[0]
    pdb_b = [
        pdb_residue_index_map[
            str(sifts_map["PDB_BEG_NUM"][i])
            + str(sifts_map["PDB_BEG_ICODE"][i])
        ]
        for i in range(index_len)
    ]
    pdb_e = [
        pdb_residue_index_map[
            str(sifts_map["PDB_END_NUM"][i])
            + str(sifts_map["PDB_END_ICODE"][i])
        ]
        + 1
        for i in range(index_len)
    ]

    for unp_b_row, unp_e_row, pdb_b_row, pdb_e_row in zip(
        unp_b, unp_e, pdb_b, pdb_e
    ):
        assert (unp_e_row - unp_b_row) == (pdb_e_row - pdb_b_row)
        assert d_mat_pdb.shape[0] >= (pdb_e_row - pdb_b_row)
        assert d_mat_unp.shape[0] >= (unp_e_row - unp_b_row)

        for unp_b_col, unp_e_col, pdb_b_col, pdb_e_col in zip(
            unp_b, unp_e, pdb_b, pdb_e
        ):
            assert (unp_e_col - unp_b_col) == (pdb_e_col - pdb_b_col)
            assert d_mat_pdb.shape[0] >= (pdb_e_col - pdb_b_col)
            assert d_mat_unp.shape[0] >= (unp_e_col - unp_b_col)
            d_mat_unp[unp_b_row:unp_e_row, unp_b_col:unp_e_col] = d_mat_pdb[
                pdb_b_row:pdb_e_row,
                pdb_b_col:pdb_e_col,
            ]

        if uniprot_seq[unp_b_row:unp_e_row] != pdb_seq[pdb_b_row:pdb_e_row]:
            warnings.warn(
                "mismatch detected between the uniprot and PDB sequence"
            )
            print("Uniprot sequence:")
            print(uniprot_seq[unp_b_row:unp_e_row])
            print("PDB sequence:")
            print(pdb_seq[pdb_b_row:pdb_e_row])

    return d_mat_unp


def get_uniprot_seq(sifts_df, d_mat_pdb_dict):
    """
    Reads from sifts_df which uniprot_id corresponds to the pdb chain in
    d_mat_dict and fetches the sequence from the web.
    """
    uniprot_id_set = set(
        sifts_df[
            (sifts_df.PDB == d_mat_pdb_dict["pdb_id"])
            & (sifts_df.CHAIN == d_mat_pdb_dict["chain_id"])
        ].SP_PRIMARY
    )
    assert len(uniprot_id_set) == 1
    uniprot_id = uniprot_id_set.pop()
    uniprot_req = requests.get(
        "https://www.uniprot.org/uniprot/" + uniprot_id + ".fasta"
    )
    fasta_as_list = uniprot_req.text.split("\n")
    header = fasta_as_list[0]
    assert header[0] == ">"
    uniprot_seq = "".join(fasta_as_list[1:])
    assert ">" not in fasta_as_list

    return uniprot_id, uniprot_seq


def get_sifts_df(sifts_csv):
    """
    Wrapper for pandas.read_csv with custom parameters
    """
    sifts_df = pd.DataFrame(
        pd.read_csv(
            sifts_csv,
            header=1,
        )
    )
    sifts_df["SP_DIFF"] = sifts_df["SP_END"] - sifts_df["SP_BEG"]
    sifts_df["RES_DIFF"] = sifts_df["RES_END"] - sifts_df["RES_BEG"]
    sifts_df["PDB_BEG_NUM"] = (
        sifts_df["PDB_BEG"].str.extract(r"(\d+)").astype(int)
    )
    sifts_df["PDB_END_NUM"] = (
        sifts_df["PDB_END"].str.extract(r"(\d+)").astype(int)
    )
    sifts_df["PDB_BEG_ICODE"] = sifts_df["PDB_BEG"].str.extract(r"([A-Za-z])")
    sifts_df["PDB_END_ICODE"] = sifts_df["PDB_END"].str.extract(r"([A-Za-z])")
    sifts_df["PDB_DIFF"] = sifts_df["PDB_END_NUM"] - sifts_df["PDB_BEG_NUM"]

    return sifts_df


def get_out_dict(d_mat_pdb_dict, sifts_df, uniprot_seq):
    """
    Map a distance matrix contained in d_mat_file using the mapping reported in
    sifts_df.
    """
    out = dict()
    # select only the current chain
    curr_df = sifts_df[
        (sifts_df.PDB == d_mat_pdb_dict["pdb_id"])
        & (sifts_df.CHAIN == d_mat_pdb_dict["chain_id"])
    ]
    sifts_map = dict()
    sifts_map["SP_BEG"] = curr_df.SP_BEG.to_numpy(dtype=int)
    sifts_map["SP_END"] = curr_df.SP_END.to_numpy(dtype=int)
    _ = sifts_map["SP_BEG"].sort(), sifts_map["SP_END"].sort()
    # these are the indexes in the pdb chain (the index reported next to each
    # atom entry, not the SEQRES index. It can include numbers and letters.
    # Because of this, I load separately the number (resseq) and letter (icode)
    # for each position.
    pdb_beg_argsort, pdb_end_argsort = (
        curr_df["PDB_BEG_NUM"].to_numpy(dtype=int).argsort(),
        curr_df["PDB_END_NUM"].to_numpy(dtype=int).argsort(),
    )
    sifts_map["PDB_BEG_NUM"] = curr_df["PDB_BEG_NUM"].to_numpy(dtype=int)[
        pdb_beg_argsort
    ]
    sifts_map["PDB_END_NUM"] = curr_df["PDB_END_NUM"].to_numpy(dtype=int)[
        pdb_end_argsort
    ]
    sifts_map["PDB_BEG_ICODE"] = curr_df["PDB_BEG_ICODE"].to_numpy(dtype=str)[
        pdb_beg_argsort
    ]
    sifts_map["PDB_END_ICODE"] = curr_df["PDB_END_ICODE"].to_numpy(dtype=str)[
        pdb_end_argsort
    ]
    out["distance_matrix"] = map_d_mat(
        d_mat_pdb_dict,
        uniprot_seq,
        sifts_map,
    )
    out["uniprot_seq"] = uniprot_seq

    return out


def main(args):
    """
    Main function
    """
    assert os.path.isfile(args.db)
    assert args.db.endswith(".csv.gz")
    assert os.path.isfile(args.i)
    assert args.i.endswith("pdb_distance_matrix.joblib.xz")
    print("Loading PDB distance matrix:", args.i)
    sifts_df = get_sifts_df(args.db)
    d_mat_pdb_dict = joblib.load(args.i)
    uniprot_id, uniprot_seq = get_uniprot_seq(sifts_df, d_mat_pdb_dict)
    out = get_out_dict(d_mat_pdb_dict, sifts_df, uniprot_seq)
    joblib.dump(
        out,
        (
            uniprot_id
            + "_mapped_"
            + d_mat_pdb_dict["pdb_id"]
            + "_"
            + d_mat_pdb_dict["chain_id"]
            + ".uniprot_distance_matrix.joblib.xz"
        ),
    )


def parse_arguments():
    """
    Parse command line arguments.
    """
    description = " ".join(__doc__.splitlines()[4:])

    epilog = ", ".join(__doc__.splitlines()[1:4])
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument(
        "-i",
        metavar="<file>",
        type=str,
        help="the output of the script ./pdb_to_distance_matrix.py",
        required=True,
    )
    parser.add_argument(
        "-db",
        metavar="<file>",
        type=str,
        help="""
        a csv file containing the uniprot to pdb mappings. It can be obtained from
        ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/uniprot_segments_observed.csv.gz
        """,
        required=True,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_arguments()
    main(ARGS)
