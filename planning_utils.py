from enum import Enum
from queue import PriorityQueue
import numpy as np
from math import sqrt
from scipy.spatial import Voronoi
from bresenham import bresenham
import numpy.linalg as LA
import networkx as nx


def create_grid_and_graph(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    along with a graph  given obstacle data and the
    drone's altitude.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min)))
    east_size = int(np.ceil((east_max - east_min)))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Center offset for grid
    north_min_center = np.min(data[:, 0])
    east_min_center = np.min(data[:, 1])

    # Define a list to hold Voronoi points
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(north - d_north - safety_distance - north_min_center),
                int(north + d_north + safety_distance - north_min_center),
                int(east - d_east - safety_distance - east_min_center),
                int(east + d_east + safety_distance - east_min_center),
            ]
            grid[obstacle[0]:obstacle[1], obstacle[2]:obstacle[3]] = alt + safety_distance - drone_altitude

            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    # create a voronoi graph based on
    # location of obstacle centres
    voronoi_graph = Voronoi(points)
    # check each edge from graph.ridge_vertices for collision
    edges = []
    for v in voronoi_graph.ridge_vertices:
        p1 = voronoi_graph.vertices[v[0]]
        p2 = voronoi_graph.vertices[v[1]]

        # check collision
        if not bres_check_collision(p1,p2,grid):
            edges.append((p1, p2))

    graph = nx.Graph()
    for edge in edges:
        dist = LA.norm(np.array(edge[0]) - np.array(edge[1]))
        graph.add_edge(tuple(edge[0]), tuple(edge[1]), weight=dist)

    return grid, graph, int(north_min), int(east_min)

def closest_point(graph, current_point):
    """
    Compute the closest point in the `graph`
    to the `current_point`.
    """
    closest_point = None
    dist = float("inf")
    for p in graph.nodes:
        d = LA.norm(np.array(p) - np.array(current_point))
        if d < dist:
            closest_point = p
            dist = d
    return closest_point


def a_star_graph(graph, heuristic, start, goal):

    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:
            print('Found a path.')
            found = True
            break

        else:
            for neighbor_node in graph[current_node]:
                if neighbor_node not in visited:
                    visited.add(neighbor_node)
                    cost = graph.edges[current_node, neighbor_node]['weight']
                    new_cost = current_cost + cost + heuristic(neighbor_node, goal)
                    queue.put((new_cost, neighbor_node))
                    branch[neighbor_node] = (new_cost, current_node)

    path = []
    path_cost = 0
    if found:
        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')

    return path[::-1], path_cost

def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def bres_check_collision(p1, p2, grid):
    """
    Note this solution requires `x1` < `x2` and `y1` < `y2`.
    """
    x1, y1 = int(np.floor(p1[0])),int(np.floor(p1[1]))
    x2, y2 = int(np.ceil(p2[0])),int(np.floor(p2[1]))
    m,n = grid.shape

    if np.min([x1, x2, y1, y2]) < 0 or max(x1, x2) > m - 1 or max(y1, y2) > n - 1:
        return True
    # TODO: Determine valid grid cells
    dy = y2 - y1
    dx = x2 - x1
    # m = np.float(y2 -y1)/(x2-x1)
    x = x1
    y = y1
    # fx+ m> y+1  convert to fx*dx +dy > y*dx+dx
    # fx = y1
    fx_dx = y1 * dx
    collision = False

    while x < x2 and y < y2:
        if grid[x][y] > 0:
            collision = True
            break

        if fx_dx + dy > y * dx + dx:
            y = y + 1
        elif fx_dx + dy < y * dx + dx:
            x = x + 1
            fx_dx += dy
        else:
            x = x + 1
            y = y + 1
            fx_dx += dy

    return collision

def point(p):
    return np.array([p[0], p[1], 1.]).reshape(1, -1)

def collinearity_check(p1, p2, p3, epsilon=1e-6):
    m = np.concatenate((p1, p2, p3), 0)
    det = np.linalg.det(m)
    return abs(det) < epsilon


def prune_path(path):
    # pruned_path = [p for p in path]
    pruned_path = []
    pruned_path.append(path[0])
    # TODO: prune the path!
    for i in range(1, len(path) - 1):
        p0 = point(pruned_path[-1])
        p1 = point(path[i])
        p2 = point(path[i + 1])
        if not collinearity_check(p0, p1, p2):
            pruned_path.append(path[i])

    pruned_path.append(path[-1])

    return pruned_path

