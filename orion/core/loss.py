import numpy as np

from orion.core import core

class CrossEntropyLoss(core.Function):
    def __init__(self, y_pred: core.Node, y_true: core.Node):
        super().__init__(input_nodes=[y_pred, y_true])
        self.probs = None
        self.y_true = None
        self.data = self.forward(y_pred.data, y_true.data)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        exp_logits = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        self.y_true = y_true

        batch_size = y_pred.shape[0]
        log_probs = -np.log(self.probs[range(batch_size), y_true] + 1e-12)
        return np.sum(log_probs) / batch_size

    def _backward(self):
        """
        反向传播计算梯度
        """
        logits = self.input_nodes[0]
        batch_size = self.probs.shape[0]

        # 计算梯度
        grad = self.probs
        grad[range(batch_size), self.y_true] -= 1
        grad /= batch_size

        # 将梯度传递回 logits
        logits.backward(grad)
