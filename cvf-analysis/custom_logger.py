import socket
import logging


from custom_mpi import program_node_rank


formatter = logging.Formatter("%(program_node_rank)s : %(host)s - %(message)s")


logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger = logging.LoggerAdapter(
    logger,
    {
        "program_node_rank": f"Node {program_node_rank}",
        "host": f"{socket.gethostname()}",
    },
)
