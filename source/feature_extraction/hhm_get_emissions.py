#!/usr/bin/python3

"""
Author: Saul Pierotti
Mail: saulpierotti.bioinfo@gmail.com
Last updated: 06/04/2021

This script takes in input a hhm file (output of hhmake of the hh-suite) and
saves the emission probabilities in a pssm, which is compressed and dumped with
joblib.
The pssm is saved as a list of dictionaries. Each position in the list is a
position in the profile, and the keys of each dictionary are the standard
aminoacids.
"""

import argparse
import os
import re

import joblib
import numpy as np


def get_pssm(file_handle):
    """
    Parse a hhm file and return a pssm
    """
    hhm_str = "".join(file_handle.readlines())
    _header, body = hhm_str.split("\n//\n")[0].split("\n#\n")
    _null_model = body.split("\n")[0:4:3]
    colnames = (
        re.sub(r"\s+", r"\t", r"\t".join(body.split("\n")[1:3]))
        .strip()
        .split()[1:-10]
    )
    values = np.array(
        [
            row[2:22] + row[23:-10]
            for row in [
                re.sub(r"\t+", "\t", re.sub(r"[\n+\s+]", r"\t", row))
                .strip()
                .split("\t")
                for row in "\n".join(body.split("\n")[4:]).split("\n\n")
            ]
        ]
    )
    values = np.where(values == "*", 99999, values).astype("int")
    pssm = [dict(zip(colnames, values_row)) for values_row in values]

    return pssm


def main(args):
    """
    Main function
    """

    assert os.path.isfile(args.i)
    assert args.i.endswith(".hhm")
    assert args.o.endswith(".hhm_pssm.joblib.xz")
    assert not os.path.isfile(args.o)

    with open(args.i) as handle:
        pssm = get_pssm(handle)

    joblib.dump(
        pssm,
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
        the hhm file
        """,
        metavar="<file>",
        required=True,
    )
    parser.add_argument(
        "-o",
        metavar="<file>",
        type=str,
        help="the file where to write the pssm",
        required=True,
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_arguments()
    main(ARGS)
