import logging
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import requests
import geopandas as gpd
from shapely.geometry import box, Polygon

from maaiveld import configure_logging
from maaiveld.datastructure import Graph, GraphManager
from maaiveld.analysis import analyze


logger = logging.getLogger(__name__)


def collect_relations(
        tile_index: dict,
        polygon_index: gpd.GeoDataFrame,
    ) -> Graph:

    graph = Graph()

    global_bbox = box(*polygon_index.total_bounds, ccw=True)
    for tile in tile_index['features']:
        tile_bbox = Polygon(*tile['geometry']['coordinates'])
        if global_bbox.intersects(tile_bbox):
            tile_id = tile['properties']['kaartbladNr']
            geoms = []
            for row in polygon_index.itertuples():
                if row.geometry.intersects(tile_bbox):
                    geoms.append((row.GPGIDENT, row.GPGNAAM, row.geometry))
            if geoms:
                url = tile['properties']['url']
                length = tile['properties']['length']
                graph.set_tile(tile_id, url=url, length=length, bbox=tile_bbox)
                for geom_id, name, geom in geoms:
                    graph.set_geom(geom_id, name=name, geometry=geom)
                    graph.set_edge(tile_id, geom_id)
    return graph


def download_tiles(graph: Graph, queue: Queue):
    for tile_id, tile_v in graph.tiles().items():
        logger.info(f"Downloading tile {tile_id} ...")
        response = requests.get(tile_v['url'])
        logger.info(f"Put download {tile_id} in queue")
        queue.put((tile_id, response))
    queue.put(None)  # Sentinel


if __name__ == "__main__":
    configure_logging()

    # ahn tile index
    tile_index_url = r'https://service.pdok.nl/rws/actueel-hoogtebestand-nederland/atom/downloads/dtm_05m/kaartbladindex.json'
    tile_index = requests.get(tile_index_url).json()

    # polygon index
    peilgebieden_vig_file = r'data/peilgebieden_vigerend.shp'
    peilgebieden_vig = gpd.read_file(peilgebieden_vig_file) #.loc[0:1, :]

    # graph of relations between tiles and peilgebieden
    graph = collect_relations(tile_index, peilgebieden_vig)
    graph.sort_tiles_by_xy()

    # parallelization of tile downloads and analysis
    download_queue = Queue(1)
    n_threads = 1
    n_procs = 4

    with GraphManager() as manager:
        graph_proxy = manager.Graph(graph)

        with ThreadPoolExecutor(max_workers=n_threads) as t_pool:
            t_pool.submit(download_tiles, graph_proxy, download_queue)

            with ProcessPoolExecutor(max_workers=n_procs, initializer=configure_logging) as p_pool:
                while True:
                    download = download_queue.get()
                    if download is None:
                        logger.info("No more downloads, exiting ...")
                        download_queue.task_done()
                        break

                    tile_id, response = download
                    logger.info(f"Dispatching download {tile_id} to processes")

                    # perform main analysis in parallel processes
                    geom_id_parts = graph_proxy.split_geom_keys_by_tile(tile_id, nparts=n_procs)
                    futures = []
                    for geom_id_part in geom_id_parts:
                        if geom_id_part:
                            futures.append(
                                p_pool.submit(
                                    analyze,            # function to execute in parallel
                                    graph_proxy,        # shared graph proxy for inter-process communication
                                    geom_id_part,       # list of geom keys to analyze in this process
                                    tile_id,            # tile id for context
                                    response.content    # tile raster data (might need shared memory at larger scale)
                                )
                            )

                    # postprocess results when all graph relations are processed
                    for f in futures:
                        try:
                            _ = f.result()
                        except Exception as e:
                            logger.error(f"Error in process: {e}")

                    download_queue.task_done()
            download_queue.join()

        graph_left = dict(graph_proxy.items())
