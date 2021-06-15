#!/usr/bin/env python3
"""
A module for parallelizing the workflow using MPI
"""

import numpy as np
from mpi4py.MPI import COMM_WORLD as comm


def worker_routine(
    get_result_chunk,
):
    """
    Each worker processes a portion of the param combinations and returns it to
    the leader
    """
    # get the chunk of param combinations to work on from the leader
    params_chunk = comm.scatter(
        None,
        root=0,
    )
    leader_vars = comm.bcast(None, root=0)
    result_chunk = get_result_chunk(params_chunk, leader_vars)
    comm.gather(result_chunk, root=0)


def leader_routine(
    preamble,
    get_params_list,
    get_result_chunk,
    conclude,
):
    """
    The leader checks the inputs, prints statements to STDOUT, collects the
    outputs from the workers and saves them to disk
    """
    leader_vars = dict()
    leader_vars = preamble(leader_vars)
    params_list = get_params_list(leader_vars)
    # split the param list in chunks and scatter them to the workers (leader
    # included)
    params_chunks_list = [
        params_chunk.tolist()
        for params_chunk in np.array_split(params_list, comm.Get_size())
    ]
    params_chunk = comm.scatter(
        params_chunks_list,
        root=0,
    )

    leader_vars = comm.bcast(leader_vars, root=0)

    # the leader acts also as worker and is the only one to collect the result
    # from the gather call

    result_chunk = get_result_chunk(params_chunk, leader_vars)

    result_chunks_list = [
        result_chunk
        for result_chunk in comm.gather(result_chunk, root=0)
        # this filters out empty chunks
        if result_chunk
    ]
    results = np.concatenate(result_chunks_list).tolist()
    conclude(params_list, results, leader_vars)
    print("*****************************************************************")
    print("Leader returned correctly")
    print("*****************************************************************")


def main(
    preamble,
    get_params_list,
    get_result_chunk,
    conclude,
):
    """
    This part is common to both leader and workers. Call rank-specific main
    function and collect exceptions. If an exception is raised anywhere, kill
    all the processes.
    """
    my_rank = comm.Get_rank()

    if my_rank == 0:
        leader_routine(preamble, get_params_list, get_result_chunk, conclude)
    else:
        worker_routine(get_result_chunk)
    print("Process rank {} terminated successfully.".format(my_rank))
