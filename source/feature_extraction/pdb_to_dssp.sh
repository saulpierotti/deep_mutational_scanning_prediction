#!/bin/bash
#
# Author: Saul Pierotti
# Mail: saulpierotti.bioinfo@gmail.com
# Last updated: 16/05/2020
#
# Takes as first argument a file containing a list of PDB IDs, or a single pdb
# file
# Processes the file(s) with the mkdssp command and create an output with the
# same name of the pdb file but extension changed to .dssp

run_dssp() {
    if [ -f "$1.cif" ]; then
        ext='cif'
    elif [ -f "$1.pdb" ]; then
        ext='pdb'
    else
        echo "Could not find input file $1 or wrong format. Aborting this input."
        return
    fi
    mkdssp -i "$1.$ext" -o "$1.dssp"
    echo "Run successfull on input $1"
}

if [ -f "$1" ]; then
    if [[ "$1" == *.txt ]]; then
        echo "Recognized main input $1 as a list of basenames."
        while read -r line; do
            run_dssp "$line"
        done <"$1"
    else
        echo "Recognized main input $1 as a structure file."
        basename=$(echo "$1" | cut -d. -f1)
        run_dssp "$basename"
    fi
else
    echo "Could not find main input file $1. Aborting everything."
    exit 1
fi
