#!/usr/bin/python3

"""
Author: Saul Pierotti
Mail: saulpierotti.bioinfo@gmail.com
Last updated: 25/05/2021

Take in input the npz output of trRosetta and produce a map of pairwise
distances.
Saves the map as a joblib dump.
"""

import argparse
import os

import joblib
import numpy as np


def get_distance_matrix(tr_rosetta_out):
    """
    Extract the distances in Ångström from the trRosetta output
    """
    tr_rosetta_mat = tr_rosetta_out["dist"]
    # protein_lenght = tr_rosetta_mat.shape[0]
    smallest_bin = 2
    bin_step = 0.5
    # first bin is binary for contact/non contact
    # a value of 1 means no contact, 0 means contact
    is_not_contact = tr_rosetta_mat[:, :, 0]
    # the remaining 36 bins are for distances from 2 to 20 Ångström with a step
    # of 0.5
    dist_bins = tr_rosetta_mat[:, :, 1:]
    is_contact = dist_bins.sum(axis=-1)
    # Number of bins, minus one for the first contact binary bin
    nbins = dist_bins.shape[-1]
    assert nbins == 36
    # Generate the midpoint of each distance bin to be able to
    # calculate correct distances. Measure in Ångström
    bin_values = np.array(
        [(smallest_bin + bin_step / 2) + bin_step * i for i in range(nbins)]
    )
    # Normalize all probabilities by dividing by the sum of the raw_predictions
    # across the last dimension, [L, L, nbins] -> [L, L, nbins]
    norm_dist_bins = np.divide(
        dist_bins, np.sum(dist_bins, axis=-1)[:, :, np.newaxis]
    )
    # Multiply normalized predictions with the bin (sizes) to get the correct value distribution
    distance_mat = np.multiply(
        norm_dist_bins, bin_values[np.newaxis, np.newaxis, :]
    ).sum(axis=-1)

    # if the summed probability of contact is lower than the probability of not
    # contact replace with +inf
    distance_mat = np.where(is_not_contact > is_contact, np.inf, distance_mat)

    return distance_mat


def get_outfile_name(npz_filename):
    """
    Produce the output filename from the input filename
    """
    basename = os.path.splitext(npz_filename)[0]

    return "{}_distance_mat.joblib.xz".format(basename)


def main(npz_filename):
    """
    The main function
    """
    outfile = get_outfile_name(npz_filename)
    print("Input file: {}".format(npz_filename))
    print("Output file: {}".format(outfile))
    tr_rosetta_out = np.load(npz_filename)
    distance_mat = get_distance_matrix(tr_rosetta_out)
    joblib.dump(distance_mat, outfile)
    print("Distance matrix saved")


def parse_arguments():
    """
    Parse command line arguments.
    """
    description = " ".join(__doc__.splitlines()[4:])
    epilog = ", ".join(__doc__.splitlines()[1:4])
    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    parser.add_argument(
        "tr_rosetta_npz",
        help="A trRosetta output file in .npz format",
        metavar="<npz_file>",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    ARGS = parse_arguments()
    main(ARGS.tr_rosetta_npz)
