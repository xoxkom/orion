import os
from queue import Queue

from graphviz import Digraph

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

    '''def view_graph(self, save_path: str):
        """
        可视化计算图，保存为指定路径的文件。
        :param save_path: 保存的文件路径，格式如 "graph.pdf" 或 "graph.png"
        """
        if not self.graph:
            raise ValueError("Cannot visualize the graph because it is empty. Add nodes to the graph first.")

        file_name = '.'.join(save_path.split('.')[:-1])
        file_format = save_path.split('.')[-1]
        dot = Digraph(format=file_format)
        dot.attr("node", style="filled")  # 设置所有节点默认填充颜色

        queue = Queue()
        visit = set()

        # 获取初始节点，假设第一个 Function 节点为起点
        for node_id in self.graph:
            node = self.id_to_node[node_id]
            if isinstance(node, core.Function):  # 找到第一个 Function 节点作为起点
                queue.put(node)
                visit.add(node)
                break

        # 遍历计算图，记录节点和边
        while not queue.empty():
            cur_node = queue.get()

            # 设置节点样式和颜色
            if isinstance(cur_node, core.Function):
                dot.node(name=cur_node.id, label=cur_node.label, fillcolor="lightgreen", shape="box")
            elif isinstance(cur_node, core.Parameter):
                dot.node(name=cur_node.id, label=cur_node.label, fillcolor="lightblue", shape="ellipse")
            elif isinstance(cur_node, core.Tensor):
                dot.node(name=cur_node.id, label=cur_node.label, fillcolor="lightpink", shape="ellipse")
            else:
                dot.node(name=cur_node.id, label=cur_node.label, fillcolor="gray", shape="ellipse")

            # 只有 Function 类型的节点有 input_nodes
            if isinstance(cur_node, core.Function):
                for input_node in cur_node.input_nodes:
                    if input_node not in visit:
                        visit.add(input_node)
                        queue.put(input_node)
                    dot.edge(input_node.id, cur_node.id)  # 使用唯一ID绘制边

        # 渲染图
        dot.render(filename=file_name, cleanup=True)
        print(f"Graph has been saved at \"{os.path.abspath(f'{file_name}.{file_format}')}\"")'''

    # noinspection PyUnresolvedReferences
    def view_graph(self, node: "core.Function", save_path: str):
        from orion.core import core
        from orion.core.tensor import Tensor

        if not self.graph:
            raise ValueError("Cannot visualize the graph because it is empty. Add nodes to the graph first.")

        file_name = '.'.join(save_path.split('.')[: -1])
        format = save_path.split('.')[-1]
        dot = Digraph(format=format)
        dot.attr("node", style="filled")  # 设置所有节点默认填充颜色

        if not isinstance(node, core.Function):
            raise ValueError("Input node must be a Function!")

        queue = Queue()
        visit = set()
        queue.put(node)
        visit.add(node)

        # 遍历计算图，记录节点和边
        while not queue.empty():
            cur_node = queue.get()

            # 设置节点样式和颜色
            if isinstance(cur_node, core.Function):
                dot.node(name=cur_node.id, label=cur_node.label, fillcolor="lightgreen", shape="box")
            elif isinstance(cur_node, core.Parameter):
                dot.node(name=cur_node.id, label=cur_node.label, fillcolor="lightblue", shape="ellipse")
            elif isinstance(cur_node, Tensor):
                dot.node(name=cur_node.id, label=cur_node.label, fillcolor="lightpink", shape="ellipse")
            else:
                dot.node(name=cur_node.id, label=cur_node.label, fillcolor="gray", shape="ellipse")

            # 只有 Function 类型的节点有 input_nodes
            if isinstance(cur_node, core.Function):
                for input_node in cur_node.input_nodes:
                    if input_node not in visit:
                        visit.add(input_node)
                        queue.put(input_node)
                    dot.edge(input_node.id, cur_node.id)  # 使用唯一ID绘制边

        # 渲染图
        dot.render(filename=file_name, cleanup=True)
        print(f"Graph has been saved at \"{os.path.abspath(f'{save_path}.{format}')}\"")

