from mpi4py import MPI

comm = MPI.COMM_WORLD
program_node_rank = comm.Get_rank()
