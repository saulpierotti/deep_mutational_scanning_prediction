#!/usr/bin/python3

"""
Author: Saul Pierotti
Mail: saulpierotti.bioinfo@gmail.com
Last updated: 07/04/2021

This script takes in input a hmm file and a master sequence (output of hmmbuild
of the HMMER package) and saves the emission probabilities mapped to the master
in a pssm, which is compressed and dumped with joblib.
The pssm is saved as a dictionary containing a numpy array with the emission
probabilities and a list of residue names (in the same order used in the
array).
"""

import argparse
import os

import joblib
import numpy as np
from Bio import SeqIO


def parse_hmm(file_handle):
    """
    Parse a hmm file and return emission probabilities for match and insert
    states, msa column reference for each match state, and the header for the
    emission columns
    """
    hmm_emissions = []
    hmm_insertions = []
    msa_columns = []
    emissions_header = None
    assert file_handle.readline().startswith("HMMER3/f")
    header = True

    for line in file_handle:
        if line.startswith("HMM "):
            header = False
            emissions_header = line.rstrip().split()[1:]
            _ = next(file_handle)  # transition line header
            _ = next(file_handle)  # compo line
            insertions_line = (
                next(file_handle).strip().split()
            )  # insert0 probabilities
            insertions_proba = insertions_line
            hmm_insertions.append(insertions_proba)
            _ = next(file_handle)  # begin to insert0 transitions

            continue

        if header:
            continue

        if line.startswith("//"):
            break
        emissions_line = line.strip().split()
        assert len(emissions_line) == 26
        insertions_line = next(file_handle).strip().split()
        _ = next(file_handle)  # transition probabilities line
        line_index = int(emissions_line[0])
        emissions_proba = emissions_line[1:21]
        insertions_proba = insertions_line
        assert len(emissions_proba) == len(emissions_proba)
        msa_column = int(emissions_line[21])  # map: column of the MSA
        _ = emissions_line[22]  # consensus residue
        _ = emissions_line[23]  # rf: reference annotation
        _ = emissions_line[24]  # mm: mask
        _ = emissions_line[25]  # cs: consensus structure
        hmm_emissions.append(emissions_proba)
        hmm_insertions.append(insertions_proba)
        msa_columns.append(msa_column)
        assert len(hmm_emissions) == line_index
    hmm_emissions = np.array(hmm_emissions, dtype="float")
    hmm_insertions = np.array(hmm_insertions, dtype="float")
    msa_columns = np.array(msa_columns)
    assert hmm_emissions.shape == (
        hmm_insertions.shape[0] - 1,
        hmm_insertions.shape[1],
    )
    assert hmm_emissions.shape[0] == msa_columns.shape[0]
    hmm_dict = {
        "emissions": hmm_emissions,
        "insertions": hmm_insertions,
        "msa_columns": msa_columns,
    }

    return hmm_dict, emissions_header


def get_pssm(hmm_dict, master):
    """
    Map the match and insert states to the master sequence and return a pssm of
    the master
    """
    pssm = []
    hmm_column = 1
    target_column = 1
    insertions_row = None
    msa_column = None

    for emissions_row, insertions_row, msa_column in zip(
        hmm_dict["emissions"],
        hmm_dict["insertions"],
        hmm_dict["msa_columns"],
    ):
        for _ in range(msa_column - target_column):
            pssm.append(insertions_row)
            target_column += 1
        assert msa_column == target_column
        pssm.append(emissions_row)
        target_column += 1
        hmm_column += 1
    hmm_column -= 1

    if msa_column is None or insertions_row is None:
        raise AssertionError

    assert np.all(
        insertions_row == hmm_dict["insertions"][-2]
    )  # there is one more insertion (insert0) than matches
    assert hmm_column == len(hmm_dict["emissions"])
    assert len(pssm) == msa_column
    end_insertion = hmm_dict["insertions"][-1]

    for _ in range(len(master) - msa_column):
        pssm.append(end_insertion)
        target_column += 1
    target_column -= 1
    assert len(master) == target_column
    assert len(master) == len(pssm)
    pssm = np.array(pssm)

    return pssm


def main(args):
    """
    Main function
    """

    assert os.path.isfile(args.i)
    assert os.path.isfile(args.m)
    assert args.i.endswith(".hmm")
    assert args.m.endswith(".fasta")
    assert args.o.endswith(".hmm_pssm.joblib.xz")
    assert not os.path.isfile(args.o)

    master = SeqIO.read(args.m, "fasta").seq

    with open(args.i) as handle:
        hmm_dict, emissions_header = parse_hmm(handle)

    pssm = get_pssm(hmm_dict, master)

    out = {"colnames": emissions_header, "pssm": pssm}

    joblib.dump(
        out,
        args.o,
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
        type=str,
        help="the hmm file in input",
        metavar="<file>",
        required=True,
    )
    parser.add_argument(
        "-m",
        type=str,
        help="the master sequence in fasta format for which to build the pssm",
        metavar="<file>",
        required=True,
    )
    parser.add_argument(
        "-o",
        metavar="<file>",
        type=str,
        help="""the file where to write the output. Must have extension
        .hmm_emissions.joblib.xz""",
        required=True,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_arguments()
    main(ARGS)
