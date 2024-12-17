# -*- encoding:utf-8 -*-
# MIT License
# Copyright (c) 2024 xoxkom
# See the LICENSE file in the project root for more details.
#
# author: xoxkom
# version: 0.1.0
# date: 2024.12.17

from abc import ABC, abstractmethod

class Module(ABC):
    """
    Base class for all neural network modules.

    Provides a framework for implementing custom modules (e.g., layers or blocks).
    It handles parameter registration and automatic child module management.

    Attributes:
        _parameters (list): List of parameters in the module.
        _modules (dict): Dictionary of sub-modules registered under this module.
    """
    def __init__(self):
        """
        Initialize a base Module with parameters and sub-modules.
        """
        self._parameters = []  # Parameters in this module
        self._modules = {}     # Child modules, registered automatically

    @abstractmethod
    def forward(self, x):
        """
        Defines the computation performed at every call.

        :param x: Input to the module.
        :return: Output after processing input.
        """
        pass

    def __call__(self, x):
        """
        Enable calling the module as a function.

        :param x: Input to the module.
        :return: Output after processing input.
        """
        return self.forward(x)

    def parameters(self):
        """
        Recursively collect all parameters from this module and its sub-modules.

        :return: List of parameters (e.g., weights and biases).
        """
        params = []
        # Collect current module's parameters
        params.extend(self._parameters)
        # Recursively collect parameters from all child modules
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def add_module(self, name, module):
        """
        Manually register a sub-module.

        :param name: Name of the sub-module.
        :param module: Sub-module to be registered.
        :raises TypeError: If the module is not an instance of Module.
        """
        if not isinstance(module, Module):
            raise TypeError(f"{name} is not an instance of Module")
        self._modules[name] = module

    def __setattr__(self, name, value):
        """
        Override setattr to automatically register child modules.

        If a value assigned to an attribute is an instance of Module, it is
        registered in the _modules dictionary.

        :param name: Name of the attribute.
        :param value: Value to assign to the attribute.
        """
        if isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)
