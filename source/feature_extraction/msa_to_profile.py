#!/usr/bin/python3

"""
Author: Saul Pierotti
Mail: saulpierotti.bioinfo@gmail.com
Last updated: 24/03/2021

This script takes in input a multiple sequence alignment in fasta
format with gaps represented as -. It produces a sequence profile
in output formatted as a numpy array and places it in a joblib
dump. The extension of the input is expected to be .aln
"""

import argparse
import os

import joblib
import numpy as np
from Bio import AlignIO
from Bio.Data import IUPACData
from sklearn.preprocessing import OrdinalEncoder


def get_profile(align_vec, enc, pseudocount=0):
    """
    Takes in input a multiple sequence alignment in a ordinally encoded numpy
    array and a fittend encoder. Produces a sequence profile in output,
    optionally with a pseudocount.
    """
    profile = []
    symbols = np.array(enc.categories_).flatten()
    assert len(symbols) == len(SYMBOLS)
    categories = enc.transform(np.expand_dims(symbols, axis=1)).flatten()
    assert np.all(categories.flatten() == range(len(SYMBOLS)))

    for column in align_vec.T:
        # add 1 instance of every possible category so that all the counts have the same lenght
        column = np.concatenate([column, categories.flatten()])
        column_categories, counts = np.unique(column, return_counts=True)
        assert np.all(column_categories == range(len(SYMBOLS)))
        # remove the artificial count and add the pseudocount
        counts = counts - 1 + pseudocount
        frequencies = counts / counts.sum()
        profile.append(frequencies)

    return np.array(profile).T, categories


def get_align_vec(align, symbols):
    """
    Return a ordinally-encoded vector representing the multiple sequence
    alignment given in input
    """
    enc = OrdinalEncoder()
    enc.fit(symbols)

    align_vec = []

    for column_in in align.T:
        column_out = enc.transform(np.expand_dims(column_in, axis=1)).flatten()
        align_vec.append(column_out)
    align_vec = np.array(align_vec).T

    return align_vec, enc


def main(args):
    """
    Main function
    """

    assert os.path.isfile(args.i)
    assert args.i.endswith(".aln")
    assert args.o.endswith("profile.joblib.xz")
    assert not os.path.isfile(args.o)

    align = np.array(AlignIO.read(args.i, "fasta"))
    align_vec, enc = get_align_vec(align, SYMBOLS)
    profile, categories = get_profile(align_vec, enc)
    profile_residue_order = enc.inverse_transform(
        np.expand_dims(categories, axis=1)
    ).flatten()
    joblib.dump(
        {
            "profile": profile,
            "residue_order": profile_residue_order,
            "num_sequences": len(align_vec),
        },
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
        help="""
        the multiple sequence alignment fasta file with extension .aln to
        use in input
        """,
        metavar="<file>",
        required=True,
    )
    parser.add_argument(
        "-o",
        metavar="<file>",
        type=str,
        help="the file where to write the profile numpy array",
        required=True,
    )

    args = parser.parse_args()

    return args


SYMBOLS = np.expand_dims(
    list(IUPACData.protein_letters + "X-"), axis=1
)

if __name__ == "__main__":
    ARGS = parse_arguments()
    main(ARGS)
