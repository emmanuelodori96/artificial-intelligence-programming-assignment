# Search Strategies:
# a. Depth First Search (DFS): DFS explores as far as possible along a branch before backtracking.
# It uses a stack for tree search.
# b. Breadth First Search (BFS): BFS explores all nodes at the current depth before moving to the next depth.
# It uses a queue for tree search.
# c. Uniform Cost Search (UCS): UCS finds the cheapest path by considering the cost of each path.
# It uses a priority queue based on path cost.
# d. Greedy Search: Greedy search selects the node that appears to be the best according to a
# heuristic function, without considering the cost to reach the node.
# e. A Search*: A* search combines cost information from UCS and heuristic information from
# greedy search to find the best path. It also uses a priority queue.

# 2.	Using sets and dictionary libraries represent the graph in Figure 1 into a python graph structure


import heapq

# Define the graph as a dictionary of dictionaries
graph = {
    'S': {'A': 3, 'B': 1},
    'A': {'B': 2, 'C': 2},
    'B': {'C': 3},
    'C': {'D': 4, 'G': 4},
    'D': {'G': 1},
    'G': {}
}

# Heuristic values for each node
heuristics = {
    'S': 7,
    'A': 5,
    'B': 7,
    'C': 4,
    'D': 1,
    'G': 0
}


class Node:
    def __init__(self, state, parent=None, action=None, cost=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

    # Add comparison methods to enable heapq
    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.cost == other.cost

# Tree Search Functions


def expand_tree(node):
    expanded_nodes = []
    if node.state in graph:
        for neighbor, cost in graph[node.state].items():
            child_node = Node(state=neighbor, parent=node,
                              action=neighbor, cost=node.cost + cost)
            expanded_nodes.append(child_node)
    return expanded_nodes


def depth_first_search_tree(start, goal):
    stack = [Node(start)]
    expanded_states = set()

    while stack:
        node = stack.pop()
        expanded_states.add(node.state)
        if node.state == goal:
            return get_solution(node), list(expanded_states), []

        stack.extend(child for child in expand_tree(node)
                     if child.state not in expanded_states)


def breadth_first_search_tree(start, goal):
    queue = [Node(start)]
    expanded_states = set()

    while queue:
        node = queue.pop(0)
        expanded_states.add(node.state)
        if node.state == goal:
            return get_solution(node), list(expanded_states), []

        queue.extend(child for child in expand_tree(node)
                     if child.state not in expanded_states)


def uniform_cost_search_tree(start, goal):
    queue = [(0, Node(start))]
    expanded_states = set()

    while queue:
        _, node = heapq.heappop(queue)
        expanded_states.add(node.state)
        if node.state == goal:
            return get_solution(node), list(expanded_states), []

        for child in expand_tree(node):
            if child.state not in expanded_states:
                heapq.heappush(queue, (child.cost, child))

# Greedy Search Function


def greedy_search_tree(start, goal):
    queue = [(heuristics[start], Node(start))]
    expanded_states = set()

    while queue:
        _, node = heapq.heappop(queue)
        expanded_states.add(node.state)
        if node.state == goal:
            return get_solution(node), list(expanded_states), []

        for child in expand_tree(node):
            if child.state not in expanded_states:
                heapq.heappush(queue, (heuristics[child.state], child))

# A* Search Function


def astar_search_tree(start, goal):
    queue = [(heuristics[start], 0, Node(start))]
    expanded_states = set()

    while queue:
        _, _, node = heapq.heappop(queue)
        expanded_states.add(node.state)
        if node.state == goal:
            return get_solution(node), list(expanded_states), []

        for child in expand_tree(node):
            if child.state not in expanded_states:
                heapq.heappush(
                    queue, (heuristics[child.state] + child.cost, child.cost, child))

# Graph Search Functions (Note: Greedy and A* are implemented for graph search as well)

# Graph Search Functions


def expand_graph(node):
    expanded_nodes = []
    if node.state in graph:
        for neighbor, cost in graph[node.state].items():
            child_node = Node(state=neighbor, parent=node,
                              action=neighbor, cost=node.cost + cost)
            expanded_nodes.append(child_node)
    return expanded_nodes


def depth_first_search_graph(start, goal):
    stack = [Node(start)]
    expanded_states = set()

    while stack:
        node = stack.pop()
        expanded_states.add(node.state)
        if node.state == goal:
            return get_solution(node), list(expanded_states), []

        stack.extend(child for child in expand_graph(node)
                     if child.state not in expanded_states)


def breadth_first_search_graph(start, goal):
    queue = [Node(start)]
    expanded_states = set()

    while queue:
        node = queue.pop(0)
        expanded_states.add(node.state)
        if node.state == goal:
            return get_solution(node), list(expanded_states), []

        queue.extend(child for child in expand_graph(node)
                     if child.state not in expanded_states)


def uniform_cost_search_graph(start, goal):
    queue = [(0, Node(start))]
    expanded_states = set()

    while queue:
        _, node = heapq.heappop(queue)
        expanded_states.add(node.state)
        if node.state == goal:
            return get_solution(node), list(expanded_states), []

        for child in expand_graph(node):
            if child.state not in expanded_states:
                heapq.heappush(queue, (child.cost, child))

# Greedy Search for Graph


def greedy_search_graph(start, goal):
    queue = [(heuristics[start], Node(start))]
    expanded_states = set()

    while queue:
        _, node = heapq.heappop(queue)
        expanded_states.add(node.state)
        if node.state == goal:
            return get_solution(node), list(expanded_states), []

        for child in expand_graph(node):
            if child.state not in expanded_states:
                heapq.heappush(queue, (heuristics[child.state], child))

# A* Search for Graph


def astar_search_graph(start, goal):
    queue = [(heuristics[start], 0, Node(start))]
    expanded_states = set()

    while queue:
        _, _, node = heapq.heappop(queue)
        expanded_states.add(node.state)
        if node.state == goal:
            return get_solution(node), list(expanded_states), []

        for child in expand_graph(node):
            if child.state not in expanded_states:
                heapq.heappush(
                    queue, (heuristics[child.state] + child.cost, child.cost, child))


# Helper function to retrieve the solution path


def get_solution(node):
    path = []
    while node:
        path.append(node.state)
        node = node.parent
    return path[::-1]

# Run the search algorithms and print results


if __name__ == "__main__":
    start_node = 'S'
    goal_node = 'G'

    # Tree Search
    dfs_tree_path, dfs_tree_expanded, _ = depth_first_search_tree(
        start_node, goal_node)
    bfs_tree_path, bfs_tree_expanded, _ = breadth_first_search_tree(
        start_node, goal_node)
    ucs_tree_path, ucs_tree_expanded, _ = uniform_cost_search_tree(
        start_node, goal_node)
    greedy_tree_path, greedy_tree_expanded, _ = greedy_search_tree(
        start_node, goal_node)
    astar_tree_path, astar_tree_expanded, _ = astar_search_tree(
        start_node, goal_node)

    # Graph Search
    dfs_graph_path, dfs_graph_expanded, _ = depth_first_search_graph(
        start_node, goal_node)
    bfs_graph_path, bfs_graph_expanded, _ = breadth_first_search_graph(
        start_node, goal_node)
    ucs_graph_path, ucs_graph_expanded, _ = uniform_cost_search_graph(
        start_node, goal_node)
    greedy_graph_path, greedy_graph_expanded, _ = greedy_search_graph(
        start_node, goal_node)
    astar_graph_path, astar_graph_expanded, _ = astar_search_graph(
        start_node, goal_node)

    # Print results
    print("Depth First Search (Tree):")
    print("Expanded States:", dfs_tree_expanded)
    print("Path:", dfs_tree_path)
    print()

    print("Breadth First Search (Tree):")
    print("Expanded States:", bfs_tree_expanded)
    print("Path:", bfs_tree_path)
    print()

    print("Uniform Cost Search (Tree):")
    print("Expanded States:", ucs_tree_expanded)
    print("Path:", ucs_tree_path)
    print()

    print("Greedy Search (Tree):")
    print("Expanded States:", greedy_tree_expanded)
    print("Path:", greedy_tree_path)
    print()

    print("A* Search (Tree):")
    print("Expanded States:", astar_tree_expanded)
    print("Path:", astar_tree_path)
    print()

    print("Depth First Search (Graph):")
    print("Expanded States:", dfs_graph_expanded)
    print("Path:", dfs_graph_path)
    print()

    print("Breadth First Search (Graph):")
    print("Expanded States:", bfs_graph_expanded)
    print("Path:", bfs_graph_path)
    print()

    print("Uniform Cost Search (Graph):")
    print("Expanded States:", ucs_graph_expanded)
    print("Path:", ucs_graph_path)
    print()

    print("Greedy Search (Graph):")
    print("Expanded States:", greedy_graph_expanded)
    print("Path:", greedy_graph_path)
    print()

    print("A* Search (Graph):")
    print("Expanded States:", astar_graph_expanded)
    print("Path:", astar_graph_path)
    print()
