import os


class Rank:
    def __init__(self, L, C, M):
        self.L = L
        self.C = C
        self.M = M

    def add_cost(self, val):
        self.L += val
        self.C += 1
        self.M = max(val, self.M)

    def __str__(self) -> str:
        return f"L: {self.L}, C: {self.C}, M: {self.M}"

    __repr__ = __str__


class ConfigurationNode:
    def __init__(self, config, children: set["ConfigurationNode"] = set()) -> None:
        self.config = config
        self.children = children  # program transitions
        self.cost = Rank(0, 0, 0)

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


graphs_dir = "graphs"
graph_names = ["small_graph_test"]


def start(graphs_dir, graph_name):
    # logger.info('Started for Graph: "%s".', graph_name)
    full_path = os.path.join(graphs_dir, f"{graph_name}.txt")
    if not os.path.exists(full_path):
        # logger.warning("Graph file: %s not found! Skipping the graph.", full_path)
        exit()

    graph = {}
    with open(full_path, "r") as f:
        line = f.readline()
        while line:
            node_edges = [int(i) for i in line.split()]
            node = node_edges[0]
            edges = node_edges[1:]
            graph[node] = set(edges)
            line = f.readline()

    return graph


class GraphColoring:
    def __init__(self) -> None:
        self.graph = start(graphs_dir, graph_names[0])
        self.nodes = list(self.graph.keys())
        self.degree_of_nodes = {n: len(self.graph[n]) for n in self.nodes}
        # self.possible_values = set(
        #     range(self.degree_of_nodes[self.nodes[position]] + 1)
        # )
        self.possible_values = []
        self.possible_values_indx = {v: i for i, v in enumerate(self.possible_values)}
        self.possible_values_indx_str = {
            v: str(i) for i, v in enumerate(self.possible_values)
        }
        self.initial_state = ConfigurationNode(tuple([0 for i in self.nodes]))

    def base_n_to_decimal(self, base_n_str):
        decimal_value = 0
        length = len(base_n_str)

        for i in range(length):
            digit = int(base_n_str[length - 1 - i])
            decimal_value += digit * (len(self.possible_values) ** i)

        return decimal_value  # base 10, not fractional value

    def config_to_indx(self, config):
        config_to_indx_str = "".join(self.possible_values_indx_str[i] for i in config)
        return self.base_n_to_decimal(config_to_indx_str)

    def start(self):
        self.find_rank()

    def _find_min_possible_color(self, colors):
        for i in range(len(colors) + 1):
            if i not in colors:
                return i

    def _is_program_transition(self, perturb_pos, start_state, dest_state):
        # if start_state in self.invariants and dest_state in self.invariants:
        #     return False
        neighbor_pos = [*self.graph[perturb_pos]]
        neighbor_colors = set(dest_state[i] for i in neighbor_pos)
        min_color = self._find_min_possible_color(neighbor_colors)

        return dest_state[perturb_pos] == min_color

    def _get_program_transitions(self, start_state):
        program_transitions = set()
        for position, val in enumerate(start_state):
            # check if node already has different color among the neighbors => If yes => no need to perturb that node's value
            # neighbor_pos = [*self.graph[position]]
            # neighbor_colors = set(start_state[i] for i in neighbor_pos)
            # if self._is_different_color(val, neighbor_colors):
            #     continue

            # if the current node's color is not different among the neighbors => search for the program transitions possible
            possible_node_colors = set(
                range(self.degree_of_nodes[self.nodes[position]] + 1)
            ) - {start_state[position]}
            for perturb_val in possible_node_colors:
                perturb_state = list(start_state)
                perturb_state[position] = perturb_val
                perturb_state = tuple(perturb_state)
                if self._is_program_transition(position, start_state, perturb_state):
                    program_transitions.add(perturb_state)

        return program_transitions

    def is_invariant(self, config):
        # return False
        for node, color in enumerate(config):
            for dest_node in self.graph[node]:
                if config[dest_node] == color:
                    return False
        return True

    def backtrack_path(self, path: list[ConfigurationNode]):
        for i, node in enumerate(path):
            node.cost.add_cost(i)

    def dfs(self, path):
        state = path[-1]
        if self.is_invariant(state.config):
            self.backtrack_path(path[::-1])
            return

        state.children = [
            ConfigurationNode(i) for i in self._get_program_transitions(state.config)
        ]
        for node in state.children:
            path_copy = path[:]
            path_copy.append(node)
            self.dfs(path_copy)

    def find_rank(self):
        self.dfs([self.initial_state])


def main():
    coloring = GraphColoring()
    coloring.start()
    coloring.initial_state.traverse()


if __name__ == "__main__":
    main()