# -*- encoding:utf-8 -*-
# MIT License
# Copyright (c) 2024 xoxkom
# See the LICENSE file in the project root for more details.
#
# author: xoxkom
# version: 0.1.0
# date: 2024.12.17

import os
from queue import Queue

from graphviz import Digraph

__all__ = [
    "Graph"
]

class _SingletonMeta(type):
    """
    A metaclass for implementing the Singleton pattern.
    Ensures that only one instance of a class is created.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Graph(metaclass=_SingletonMeta):
    """
    A singleton class representing the computation graph.

    The Graph class manages all nodes and their relationships within the
    computation graph. It supports appending nodes, clearing the graph,
    and visualizing the graph using Graphviz.

    Attributes:
        graph (list): List of node IDs in the graph.
        id_to_node (dict): Mapping of node IDs to their corresponding node objects.
    """
    def __init__(self):
        """
        Initialize an empty computation graph.
        """
        self.graph = []
        self.id_to_node = {}

    def append(self, node):
        """
        Append a node to the computation graph.

        :param node: The node to be added to the graph.
        """
        self.graph.append(node.id)
        self.id_to_node[node.id] = node

    def get_graph(self):
        """
        Get the list of node IDs in the graph.

        :return: List of node IDs.
        """
        return self.graph

    def get_node(self, node_id):
        """
        Retrieve a node from the graph using its ID.

        :param node_id: The ID of the node to retrieve.
        :return: The corresponding node object, or None if not found.
        """
        return self.id_to_node.get(node_id)

    def clear(self):
        """
        Clear all nodes and relationships from the computation graph.
        """
        self.graph.clear()
        self.id_to_node.clear()

    # noinspection PyUnresolvedReferences
    def view_graph(self, node: "core.Function", save_path: str):
        """
        Visualize the computation graph using Graphviz.

        :param node: A starting node (must be an instance of core.Function).
        :param save_path: The file path to save the rendered graph.
        :raises ValueError: If the graph is empty or the input node is invalid.
        """
        from orion.core import core
        from orion.tensor.tensor import Tensor

        if not self.graph:
            raise ValueError("Cannot visualize the graph because it is empty. Add nodes to the graph first.")

        file_name = '.'.join(save_path.split('.')[: -1])
        format = save_path.split('.')[-1]
        dot = Digraph(format=format)
        dot.attr("node", style="filled")  # Set default node style to filled

        if not isinstance(node, core.Function):
            raise ValueError("Input node must be a Function!")

        queue = Queue()
        visit = set()
        queue.put(node)
        visit.add(node)

        # Traverse the computation graph and record nodes and edges
        while not queue.empty():
            cur_node = queue.get()

            # Set node styles and colors
            if isinstance(cur_node, core.Function):
                dot.node(name=cur_node.id, label=cur_node.label, fillcolor="lightgreen", shape="box")
            elif isinstance(cur_node, core.Parameter):
                dot.node(name=cur_node.id, label=cur_node.label, fillcolor="lightblue", shape="ellipse")
            elif isinstance(cur_node, Tensor):
                dot.node(name=cur_node.id, label=cur_node.label, fillcolor="lightpink", shape="ellipse")
            else:
                dot.node(name=cur_node.id, label=cur_node.label, fillcolor="gray", shape="ellipse")

            # Only Function nodes have input_nodes
            if isinstance(cur_node, core.Function):
                for input_node in cur_node.input_nodes:
                    if input_node not in visit:
                        visit.add(input_node)
                        queue.put(input_node)
                    dot.edge(input_node.id, cur_node.id)  # Add an edge between nodes

        # Render and save the graph
        dot.render(filename=file_name, cleanup=True)
        print(f"Graph has been saved at \"{os.path.abspath(f'{save_path}')}\"")
