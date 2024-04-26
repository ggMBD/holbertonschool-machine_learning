#!/usr/bin/env python3
"""0. Depth of a decision tree """
import numpy as np


class Node:
    """node class for a decision tree"""
    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        """constructor for the node class"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """returns the depth of the node"""
        if self.is_leaf:
            return self.depth
        else:
            return max(self.left_child.max_depth_below(),
                       self.right_child.max_depth_below())


class Leaf(Node):
    """leaf class for a decision tree"""
    def __init__(self, value, depth=None):
        """constructor for the leaf class"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """returns the depth of the leaf node"""
        return self.depth


class Decision_Tree():
    """decision tree class"""
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """constructor for the decision tree class"""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """returns the depth of the decision tree"""
        return self.root.max_depth_below()
