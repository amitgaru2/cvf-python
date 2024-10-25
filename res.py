class TreeNode:
    def __init__(self, config, children: set["TreeNode"] = set()) -> None:
        self.config = config
        self.children = children
        self.cost = None

    def traverse(self):
        queue = [self]
        while queue:
            operating_node = queue.pop(0)
            print(operating_node)
            for child in operating_node.children:
                queue.append(child)

    def __str__(self) -> str:
        return f"{self.config} : {self.cost}"

    __repr__ = __str__


def get_pts(config):
    return {
        (0, 0, 0): {(0, 0, 1)},
        (0, 0, 1): {(0, 1, 0), (0, 1, 1)},
        (0, 1, 0): set(),
        (0, 1, 1): {(0, 1, 0)},
    }[config]


def dfs(path):
    state = path[-1]
    if state.config in invariants:
        backtrack_path(path[::-1])
        return

    state.children = [TreeNode(i) for i in get_pts(state.config)]
    for node in state.children:
        path_copy = path[:]
        path_copy.append(node)
        dfs(path_copy)


def backtrack_path(path: list[TreeNode]):
    for i, node in enumerate(path):
        node.cost = (i + node.cost) / 2 if node.cost is not None else i


invariants = {(0, 1, 0)}
start_state = TreeNode((0, 0, 0))

dfs([start_state])

start_state.traverse()
