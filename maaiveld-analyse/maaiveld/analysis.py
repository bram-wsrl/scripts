import time
import logging
import datetime as dt

from maaiveld.datastructure import Graph
# import rioxarray as rxr


def analyze(graph: Graph, geom_keys: list, tile_id: str, tile_data: bytes):
    logging.info(f"{dt.datetime.now()} Analyzing data for tile {tile_id}...")
    for geom_key in geom_keys:
        logging.info(f"Analyzing geom {geom_key} for tile {tile_id}...")
        time.sleep(1)

        graph.set_edge(tile_id, geom_key, visited=True)

        logging.info(f"Finished analyzing geom {geom_key} for tile {tile_id} ...")

    logging.info(f"{dt.datetime.now()} Finished analyzing data for tile {tile_id}.")
    return
