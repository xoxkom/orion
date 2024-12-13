from __future__ import annotations

import os
from abc import abstractmethod
from queue import Queue

from graphviz import Digraph
import numpy as np

from orion.core.graph import Graph
from orion.core.config import get_root_path

__all__ = [
    "Node",
    "Function",
    "Tensor",
    "Parameter",
    "Negative",
    "Add",
    "Minus",
    "Multiply",
    "Matmul",
    "view_graph"
]

root_path = get_root_path()

_default_graph = Graph()

class Node(object):
    """ Base for all types of node in the static calculation graph. """
    _global_id_counter = {}

    def __init__(self):
        self.next_nodes = []
        self.data = None
        self.label = self.__class__.__name__

        class_name = self.__class__.__name__
        if class_name not in Node._global_id_counter:
            Node._global_id_counter[class_name] = 0
        Node._global_id_counter[class_name] += 1
        self.id = f"{class_name}_{Node._global_id_counter[class_name]}"

        global _default_graph
        _default_graph.append(self)

    def __neg__(self):
        return Negative(self)

    def __add__(self, other: Node):
        return Add(self, other)

    def __sub__(self, other: Node):
        return Minus(self, other)

    def __mul__(self, other: Node):
        return Multiply(self, other)

    def __matmul__(self, other: Node):
        return Matmul(self, other)

    def __str__(self):
        return f"{self.__class__.__name__}({str(self.data)})"

    def __repr__(self):
        return str(self)

    @property
    def numpy(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the forward method first!")
        return np.array(self.data)

    @property
    def shape(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the forward method first!")
        return np.array(self.data).shape

    def to_numpy(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the forward method first!")
        return np.array(self.data)

    def to_list(self):
        if self.data is None:
            raise ValueError(f"the node {self.__class__} is empty, please run the forward method first!")
        return list(self.data)

class Function(Node):
    def __init__(self, input_nodes: list[Node]):
        super().__init__()
        self.input_nodes = input_nodes
        for node in input_nodes:
            node.next_nodes.append(self)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass


class Tensor(Node):
    def __init__(self, data: np.ndarray):
        super().__init__()
        if not isinstance(data, np.ndarray):
            raise TypeError("Input data must be a NumPy ndarray.")

        self.data = data

    @property
    def shape(self):
        """Return the shape of the tensor."""
        return self.data.shape

    def __str__(self):
        return f"Tensor(shape={self.shape}, data={self.data})"

class Parameter(Node):
    def __init__(self,
                 init_value: np.ndarray | list = None,
                 shape: tuple[int, ...] = None,
                 initializer: str = "default",
                 label: str = None):
        super().__init__()
        if init_value is not None:
            self.data = np.array(init_value, dtype=np.float32)
        elif shape is not None:
            if initializer == "default":
                self.data = np.random.uniform(-0.1, 0.1, size=shape).astype(np.float32)
            else:
                raise ValueError(f"Unsupported initializer: {initializer}")
        else:
            raise ValueError("Either `init_value` or `shape` must be provided.")

        if label is not None:
            self.label = label
        self._transposed = None  # Cache for transposed object

class Add(Function):
    def __init__(self, x: Node, y: Node):
        super().__init__(input_nodes=[x, y])
        self.data = self.forward(x.data, y.data)

    def forward(self, x: np.ndarray, y: np.ndarray):
        return x + y

class Negative(Function):
    def __init__(self, x: Node):
        super().__init__(input_nodes=[x])
        self.data = self.forward(x.data)

    def forward(self, x: np.ndarray):
        return -1. * x


class Minus(Function):
    def __init__(self, x: Node, y: Node):
        super().__init__(input_nodes=[x, y])


    def forward(self, x: np.ndarray, y: np.ndarray):
        return x - y

class Multiply(Function):
    def __init__(self, x: Node, y: Node):
        super().__init__(input_nodes=[x, y])
        self.data = self.forward(x.data, y.data)

    def forward(self, x: np.ndarray, y: np.ndarray):
        return x * y

class Matmul(Function):
    def __init__(self, x: Node, y: Node):
        super().__init__(input_nodes=[x, y])
        self.data = self.forward(x.data, y.data)

    def forward(self, x: np.ndarray, y: np.ndarray):
        return x @ y

def view_graph(node: Function, file_name: str=f"{root_path}/tmp/dot", format="png"):
    dot = Digraph(format=format)
    dot.attr("node", style="filled")  # 设置所有节点默认填充颜色

    if not isinstance(node, Function):
        raise ValueError("Input node must be a Function!")

    queue = Queue()
    visit = set()
    queue.put(node)
    visit.add(node)

    # 遍历计算图，记录节点和边
    while not queue.empty():
        cur_node = queue.get()

        # 设置节点样式和颜色
        if isinstance(cur_node, Function):
            dot.node(name=cur_node.id, label=cur_node.label, fillcolor="lightgreen", shape="box")
        elif isinstance(cur_node, Parameter):
            dot.node(name=cur_node.id, label=cur_node.label, fillcolor="lightblue", shape="ellipse")
        elif isinstance(cur_node, Tensor):
            dot.node(name=cur_node.id, label=cur_node.label, fillcolor="lightpink", shape="ellipse")
        else:
            dot.node(name=cur_node.id, label=cur_node.label, fillcolor="gray", shape="ellipse")

        # 只有 Function 类型的节点有 input_nodes
        if isinstance(cur_node, Function):
            for input_node in cur_node.input_nodes:
                if input_node not in visit:
                    visit.add(input_node)
                    queue.put(input_node)
                dot.edge(input_node.id, cur_node.id)  # 使用唯一ID绘制边

    # 渲染图
    dot.render(filename=file_name, cleanup=True)
    print(f"Graph has been saved at \"{os.path.abspath(f'{file_name}.{format}')}\"")

