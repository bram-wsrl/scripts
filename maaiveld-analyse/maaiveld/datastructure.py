import itertools as it
from multiprocessing.managers import BaseManager


class Graph(dict):
    """
    graph = {
        tiles: {
            tile_id0: {},
            },
        geoms: {
            'geom_id0': {},
            'geom_id1': {},
        },
        edges: {
            'tile_id0': {
                'geom_id0': {},
                'geom_id1': {},
            },
        }
    }
    """
    def repr(self):
        return (
            f'{self.__class__.__name__}('
            f'tiles={len(self.tiles())}, '
            f'geoms={len(self.geoms())}, '
            f'edges={len(self.edges().keys())})')

    def tiles(self) -> dict:
        return self.setdefault('tiles', {})

    def geoms(self) -> dict:
        return self.setdefault('geoms', {})
    
    def edges(self) -> dict:
        return self.setdefault('edges', {})

    def tile(self, tile_id) -> dict:
        return self.tiles()[tile_id]

    def set_tile(self, tile_id, **kwargs):
        value = self.tiles().get(tile_id, {})
        value.update(kwargs or {})
        self.tiles()[tile_id] = value

    def set_geom(self, geom_id, **kwargs):
        value = self.geoms().get(geom_id, {})
        value.update(kwargs or {})
        self.geoms()[geom_id] = value

    def set_edge(self, tile_id, geom_id, **kwargs):
        tile_edges = self.edges().get(tile_id, {})
        value = tile_edges.get(geom_id, {})
        value.update(kwargs or {})
        tile_edges[geom_id] = value
        self.edges()[tile_id] = tile_edges

    def sort_tiles_by_xy(self):
        tile_ids = sorted(
            self.tiles().keys(),
            key=lambda item: (
                self.tiles()[item]['bbox'].bounds[0],
                self.tiles()[item]['bbox'].bounds[1]
                ),
            reverse=False
        )
        self.tiles().update({tile_id: self.tiles()[tile_id] for tile_id in tile_ids})

    def split_geom_keys_by_tile(self, tile_id: str, nparts: int = 1):
        tile_edges = self.edges().get(tile_id, {})
        geom_ids = list(tile_edges.keys())
        k, m = divmod(len(geom_ids), nparts)
        lst = iter(geom_ids)
        geom_id_parts = [list(it.islice(lst, k + (i < m))) for i in range(nparts)]
        return geom_id_parts


class GraphManager(BaseManager):
    pass

GraphManager.register(
    'Graph',
    Graph
)
