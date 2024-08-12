import networkx as nx
import random

def random_insert_null_op(graph):
    pass

def random_commute(graph):
    pass

TRANSFORM_MAP = {
    'random_insert_null_op': random_insert_null_op,
    'random_commute': random_commute
}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, graph):
        for t in self.transforms:
            graph = t(graph)
        return graph
    

def perform_random_transform(graph, transforms, compose=False):
    """
    Perform a random transformation (also composite) on the input graph.
    """
    if transforms is None:
        # Perform a random transformation among the ones defined in this script
        transforms = TRANSFORM_MAP.values()
    else:
        for t in transforms:
            if t not in TRANSFORM_MAP:
                raise ValueError(f"Transform '{t}' is not implemented. Available transforms: {list(TRANSFORM_MAP.keys())}.")
        transforms = [TRANSFORM_MAP[t] for t in transforms]
    
    if compose:
        transform = Compose(random.choice(transforms, 2, replace=True))
    else:
        transform = random.choice(transforms)
    return transform(graph)