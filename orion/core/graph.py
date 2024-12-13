class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Graph(metaclass=SingletonMeta):
    def __init__(self):
        self.graph = []
        self.id_to_node = {}

    def append(self, node):
        self.graph.append(node.id)
        self.id_to_node[node.id] = node

    def get_graph(self):
        return self.graph

    def get_node(self, node_id):
        return self.id_to_node.get(node_id)
