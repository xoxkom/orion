from orion.core import act, core

class Linear(object):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 initializer: str = "default",
                 label: str = None):
        if label is not None:
            self.label = label
        else:
            self.label = "Linear"

        # self.data = None
        self.weight = core.Parameter(shape=(in_features, out_features), initializer=initializer, label=f"{self.label}.weight")
        self.bias = core.Parameter(shape=(out_features,), initializer="default", label=f"{self.label}.bias") if bias else None

        self.in_features = in_features
        self.out_features = out_features

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])

    def forward(self, x: core.Node):
        if x.shape[1] != self.in_features:
            raise ValueError(f"Input shape {x.shape[1]} does not match in_features {self.in_features}")

        y = x @ self.weight
        if self.bias is not None:
            y = y + self.bias
        return y

    def backward(self):
        pass

class ReLU(object):
    def __init__(self, label: str = None):
        if label is not None:
            self.label = label
        else:
            self.label = "ReLU"

    def forward(self, x: core.Node):
        return act.ReLU(x)

    def backward(self):
        pass
