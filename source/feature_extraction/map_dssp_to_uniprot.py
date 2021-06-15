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

import joblib
import numpy as np
import pandas as pd
import requests

RESIDUE_MAX_ACC = {
    # this is the ASA for the residue in an extended Ala-X-Ala trypeptide
    # Values from Ahmad et al. 2003, https://doi.org/10.1002/prot.10328
    "A": 110.2,
    "R": 229.0,
    "N": 146.4,
    "D": 144.1,
    "C": 140.4,
    "Q": 178.6,
    "E": 174.7,
    "G": 78.70,
    "H": 181.9,
    "I": 183.1,
    "L": 164.0,
    "K": 205.7,
    "M": 200.1,
    "F": 200.7,
    "P": 141.9,
    "S": 117.2,
    "T": 138.7,
    "W": 240.5,
    "Y": 213.7,
    "V": 153.7,
}


def parse_dssp(dssp_file, chain_id):
    """
    Takes in input a dssp output file and which chain I am interested in.
    Creates a pandas dataframe with all the data.
    The chain is the author one, not the DSSP one.
    """
    as_dict = {
        "pdb_res": [],
        "chain_id": [],
        "residue_type": [],
        "secondary_structure": [],
        "solvent_accessibility": [],
        "phi": [],
        "psi": [],
    }
    header_read = False
    curr_chain = "Initialised value"
    with open(dssp_file) as handle:
        for line in handle:
            if line.lstrip().startswith("#"):
                header_read = True

                continue

            if not header_read:
                continue
            # I use the authchain field, not the dssp chain

            if len(line) > 162:
                curr_chain = line[162]

            if curr_chain != chain_id:
                continue

            if line[13:15].strip() == "!*":
                break
            elif line[13:15].strip() == "!":
                continue
            # can contain icode
            as_dict["pdb_res"].append(line[5:10].strip())
            # I use the authchain field, not the dssp chain
            as_dict["chain_id"].append(line[162])
            as_dict["residue_type"].append(line[13])
            as_dict["secondary_structure"].append(line[16])
            as_dict["solvent_accessibility"].append(float(line[34:38]))
            as_dict["phi"].append(float(line[103:109]))
            as_dict["psi"].append(float(line[109:115]))
    as_df = pd.DataFrame(as_dict)
    as_df["secondary_structure"].replace(" ", "C", inplace=True)
    # disulfide bridges are small letters, so I make them cisteine
    as_df["residue_type"] = as_df["residue_type"].str.replace(
        r"([a-z])", "C", regex=True
    )
    as_df["maximum_solvent_accessibility"] = [
        RESIDUE_MAX_ACC[residue] for residue in as_df["residue_type"]
    ]
    as_df["relative_solvent_accessibility"] = (
        as_df["solvent_accessibility"] / as_df["maximum_solvent_accessibility"]
    )

    return as_df


def map_dssp(dssp_df, sifts_map):
    """
    Add a uniprot_id column to dssp_df with a reference to the corresponding
    Uniprot position.
    """
    # get pdb indexes from the resseq and icode values
    pdb_residue_index_map = {
        str(pdb_res): i for i, pdb_res in enumerate(dssp_df["pdb_res"])
    }
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
    sifts_map["PDB_BEG_ICODE"] = np.where(
        sifts_map["PDB_BEG_ICODE"] == "nan", "", sifts_map["PDB_BEG_ICODE"]
    )
    sifts_map["PDB_END_ICODE"] = np.where(
        sifts_map["PDB_END_ICODE"] == "nan", "", sifts_map["PDB_END_ICODE"]
    )
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

    pdb_index_range = np.concatenate(
        [range(b, e) for b, e in zip(pdb_b, pdb_e)]
    )
    uniprot_res_index = pd.Series(
        dtype=int, index=pdb_index_range, name="uniprot_res_index"
    )

    for curr_unp_b, curr_unp_e, curr_pdb_b, curr_pdb_e in zip(
        unp_b, unp_e, pdb_b, pdb_e
    ):
        assert (curr_unp_e - curr_unp_b) == (curr_pdb_e - curr_pdb_b)
        assert len(dssp_df) >= (curr_pdb_e - curr_pdb_b)
        assert len(dssp_df) >= (curr_unp_e - curr_unp_b)
        # loc includes also the stop index
        uniprot_res_index.loc[curr_pdb_b : curr_pdb_e - 1] = range(
            curr_unp_b, curr_unp_e
        )

    # I use inner join since I want to drop PDB positions that do not map to
    # UniProt
    dssp_df_mapped = dssp_df.join(uniprot_res_index, how="inner")
    dssp_df_mapped["uniprot_res_pos"] = dssp_df_mapped["uniprot_res_index"] + 1

    if (
        len(dssp_df) != len(dssp_df_mapped)
        # second comparison raises an error if the lenghts are different!
        or (dssp_df_mapped.index != dssp_df.index).all()
    ):
        unmapped_idx = dssp_df.index.difference(dssp_df_mapped.index)
        print(
            "Warning: Some positions were dropped from the PDB since they did \
not map to UniProt. Affected DSSP rows:"
        )
        print(dssp_df.loc[unmapped_idx])

    return dssp_df_mapped


def get_uniprot_seq(sifts_df, pdb_code, pdb_chain):
    """
    Reads from sifts_df which uniprot_id corresponds to pdb_code
    and fetches the sequence from the web.
    """
    uniprot_id_set = set(
        sifts_df[
            (sifts_df.PDB == pdb_code) & (sifts_df.CHAIN == pdb_chain)
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


def get_out_df(dssp_df, sifts_df, pdb_id, chain_id):
    """
    Map the dssp dictionary to the uniprot sequence using the mapping reported
    in sifts_df.
    """
    # select only the current chain
    curr_df = sifts_df[(sifts_df.PDB == pdb_id) & (sifts_df.CHAIN == chain_id)]
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

    return map_dssp(dssp_df, sifts_map)


def main(args):
    """
    Main function
    """
    assert os.path.isfile(args.db)
    assert args.db.endswith(".csv.gz")
    assert os.path.isfile(args.i)
    assert args.i.endswith(".dssp")
    print("Loading DSSP file:", args.i)
    print("Assuming PDB code {} and chain id {}".format(args.pdb, args.chain))
    sifts_df = get_sifts_df(args.db)
    dssp_df = parse_dssp(args.i, args.chain)
    uniprot_id, _ = get_uniprot_seq(sifts_df, args.pdb, args.chain)
    out = get_out_df(dssp_df, sifts_df, args.pdb, args.chain)
    out_file = "{}_mapped_{}_{}.uniprot_dssp.joblib.xz".format(
        uniprot_id, args.pdb, args.chain
    )
    joblib.dump(
        out,
        out_file,
    )
    print("Saved successfully to", out_file)


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
        help="the output of mkdssp",
        required=True,
    )
    parser.add_argument(
        "-pdb",
        metavar="<id>",
        type=str,
        help="the PDB ID of the structure used to obtain the DSSP file",
        required=True,
    )
    parser.add_argument(
        "-chain",
        metavar="<id>",
        type=str,
        help="the PDB chain of interest",
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
