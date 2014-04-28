"""
Implements a dependency tree. Indices are 1-based.
"""

class Node:
    def __init__(self, pair, index):
        self.parent_index_, self.label_ = pair.split('/')
        self.parent_index_ = int(self.parent_index_)
        self.index_ = int(index)
        self.parent_ = None
        self.children_ = []

    def parent(self):
        return (self.parent_, self.label_)

    def parent_index(self):
        return self.parent_index_

    def index(self):
        return self.index_

    def label(self):
        """Label of the arc pointing to the node's parent."""
        return self.label_

    def children(self):
        return self.children_

    def __str__(self):
        return '-%s/%d->' % (self.label_, self.parent_index_)

class DepTree:
    def __init__(self, line):
        self.nodes_ = [Node('-1/None', 0)] + [Node(x,i+1) for i,x in enumerate(line.rstrip().split())]
        for node in self.nodes_[1:]:
            self.nodes_[node.parent_index()].children_.append(node)
            node.parent = node.parent_index()

    def root(self):
        return self.node(0)

    def node(self, index):
        return self.nodes_[index]

    def nodes(self):
        return self.nodes_

    def __iter__(self):
        """Iterate over the nodes from left to right."""
        self.iter_node_ = 1
        return self

    def next(self):
        if self.iter_node_ >= len(self.nodes_):
            raise StopIteration
        else:
            self.iter_node_ += 1
            return self.nodes_[self.iter_node_ - 1]
