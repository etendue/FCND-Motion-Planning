from enum import Enum
from queue import PriorityQueue
import numpy as np
from math import sqrt
from scipy.spatial import Voronoi
from bresenham import bresenham
import numpy.linalg as LA
import networkx as nx


class Action(Enum):
    """
    An action is represented by a 3 element tuple.
    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    # here the definition is wrong, in execise, it is up/down,left/right, which is grid coordinate,
    # here we define the geo coordinate so, it must confirm to geo rule.
    NORTH = (1, 0, 1)
    SOUTH = (-1, 0, 1)
    # add diagonal actions
    NORTH_EAST = (1, 1, sqrt(2))
    NORTH_WEST = (1, -1, sqrt(2))
    SOUTH_EAST = (-1, 1, sqrt(2))
    SOUTH_WEST = (-1, -1, sqrt(2))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    # code can be optimized a little bit
    # check the relative altitude with drone
    if x - 1 < 0 or grid[x - 1, y] > 0:
        valid_actions.remove(Action.SOUTH)
    if x + 1 > n or grid[x + 1, y] > 0:
        valid_actions.remove(Action.NORTH)
    if y - 1 < 0 or grid[x, y - 1] > 0:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] > 0:
        valid_actions.remove(Action.EAST)
    # extend check for diagonal action
    if x - 1 < 0 or y - 1 < 0 or grid[x - 1, y - 1] > 0:
        valid_actions.remove(Action.SOUTH_WEST)
    if x + 1 > n or y - 1 < 0 or grid[x + 1, y - 1] > 0:
        valid_actions.remove(Action.NORTH_WEST)
    if x - 1 < 0 or y + 1 > m or grid[x - 1, y + 1] > 0:
        valid_actions.remove(Action.SOUTH_EAST)
    if x + 1 > n or y + 1 > m or grid[x + 1, y + 1] > 0:
        valid_actions.remove(Action.NORTH_EAST)

    return valid_actions


def a_star_grid(grid, h, start, goal):
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set()
    visited.add(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:
            current_cost = branch[current_node][0]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
        return path[::-1], path_cost
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
        return None, None


def a_star_graph(graph, heuristic, start, goal):
    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set()
    visited.add(start)
    branch = {}
    branch[start] = (heuristic(start, goal), None)
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

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
        return path[::-1], path_cost
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')
        return None, None


def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def ray_tracing_bresham(p1, p2, grid):
    """
    check if line between p1 and p2 is blocked
    return True if line is not blocked
    """

    m, n = grid.shape
    # make x1 <=x2, switch p1,p2 when necessary
    x1, y1, x2, y2 = (p1[0], p1[1], p2[0], p2[1]) if p1[0] < p2[0] else (p2[0], p2[1], p1[0], p1[1])
    min_y, max_y = (y1, y2) if y1 < y2 else (y2, y1)

    if y2 > y1:
        x1, y1 = int(np.floor(x1)), int(np.floor(y1))
        x2, y2 = int(np.ceil(x2)), int(np.ceil(y2))
    else:
        x1, y1 = int(np.floor(x1)), int(np.ceil(y1))
        x2, y2 = int(np.ceil(x2)), int(np.floor(y2))

    if x1 < 0 or x2 > m - 1 or min_y < 0 or max_y > n - 1:
        return False

    dy = y2 - y1
    dx = x2 - x1
    # m = np.float(y2 -y1)/(x2-x1)
    # fx+ m> y+1  convert to fx*dx +dy > y*dx+dx
    # fx = y1

    collision = False
    x, y = (x1, y1)
    fx_dx = y * dx

    if dy > 0:
        while x <= x2 and y <= y2:
            if grid[x][y] > 0:
                collision = True
                break

            if fx_dx + dy > y * dx + dx:
                y += 1
            elif fx_dx + dy < y * dx + dx:
                fx_dx += dy
                x += 1
            else:
                x += 1
                y += 1
                fx_dx += dy
    elif dy < 0:
        while x <= x2 and y >= y2:
            if grid[x][y - 1] > 0:
                collision = True
                break

            if fx_dx + dy > y * dx - dx:
                fx_dx += dy
                x += 1
            elif fx_dx + dy < y * dx - dx:
                y -= 1
            else:
                x += 1
                y -= 1
                fx_dx += dy
    else:
        while x <= x2:
            if grid[x][y] > 0:
                collision = True
                break
            x += 1

    return not collision


def bresham(p1, p2):
    """
    return a list of cells with integer index
    """

    # make x1 <=x2, switch p1,p2 when necessary
    reverse = p1[0] > p2[0]
    x1, y1, x2, y2 = (p1[0], p1[1], p2[0], p2[1]) if not reverse else (p2[0], p2[1], p1[0], p1[1])
    min_y, max_y = (y1, y2) if y1 < y2 else (y2, y1)

    if y2 > y1:
        x1, y1 = int(np.floor(x1)), int(np.floor(y1))
        x2, y2 = int(np.ceil(x2)), int(np.ceil(y2))
    else:
        x1, y1 = int(np.floor(x1)), int(np.ceil(y1))
        x2, y2 = int(np.ceil(x2)), int(np.floor(y2))

    dy = y2 - y1
    dx = x2 - x1
    # m = np.float(y2 -y1)/(x2-x1)
    # fx+ m> y+1  convert to fx*dx +dy > y*dx+dx
    # fx = y1

    x, y = (x1, y1)
    fx_dx = y * dx
    cells = []
    if dy > 0:
        while x <= x2 and y <= y2:
            cells.append((x, y))
            if fx_dx + dy > y * dx + dx:
                y += 1
            elif fx_dx + dy < y * dx + dx:
                fx_dx += dy
                x += 1
            else:
                x += 1
                y += 1
                fx_dx += dy
    elif dy < 0:
        while x <= x2 and y >= y2:
            cells.append((x, y - 1))
            if fx_dx + dy > y * dx - dx:
                fx_dx += dy
                x += 1
            elif fx_dx + dy < y * dx - dx:
                y -= 1
            else:
                x += 1
                y -= 1
                fx_dx += dy
    else:
        while x <= x2:
            cells.append((x, y))
            x += 1
    if reverse:
        cells = cells[::-1]
    return cells


def collinearity_check(p1, p2, p3, epsilon=1e-6):
    m = np.concatenate((p1, p2, p3), 0)
    det = np.linalg.det(m)
    return abs(det) < epsilon

def prune_path_collinear(path):
    # pruned_path = [p for p in path]
    pruned_path = []
    pruned_path.append(path[0])
    for i in range(1, len(path) - 1):
        p0 = np.array(pruned_path[-1]).reshape(1,-1)
        p1 = np.array(path[i]).reshape(1,-1)
        p2 = np.array(path[i + 1]).reshape(1,-1)
        if not collinearity_check(p0, p1, p2):
            pruned_path.append(path[i])

    pruned_path.append(path[-1])

    return pruned_path


def prune_path_ray_tracing(path, grid):
    pruned_path = []
    pruned_path.append(path[0])
    # remove intermediate points
    for i in range(1, len(path) - 1):
        p1 = pruned_path[-1]
        p2 = path[i]
        p3 = path[i + 1]
        if not ray_tracing_bresham(p1, p3, grid) or p2[2] != p1[2]:
            pruned_path.append(p2)

    pruned_path.append(path[-1])
    return pruned_path


def create_grid(data, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size), dtype=np.float)

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        obstacle = [
            int(np.clip(north - d_north - safety_distance - north_min, 0, north_size - 1)),
            int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
            int(np.clip(east - d_east - safety_distance - east_min, 0, east_size - 1)),
            int(np.clip(east + d_east + safety_distance - east_min, 0, east_size - 1)),
        ]
        grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = alt + d_alt

    return grid, int(north_min), int(east_min)


def create_graph(block_centers, grid):
    """
    Returns a grid representation of a 2D configuration space
    along with a graph  given obstacle data and the
    drone's altitude.
    """
    # create a voronoi graph based on
    # location of obstacle centres
    vi = Voronoi(block_centers)
    graph = nx.Graph()
    for vertex in vi.ridge_vertices:
        p1, p2 = vi.vertices[vertex[0]], vi.vertices[vertex[1]]
        # check each edge from graph.ridge_vertices for collision
        if ray_tracing_bresham(p1, p2, grid):
            dist = LA.norm(np.array(p1) - np.array(p2))
            graph.add_edge((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), weight=dist)

    # find the biggest subgraph which is connected.
    components = list(nx.connected_component_subgraphs(graph))
    number_of_nodes = [com.number_of_nodes() for com in components]
    subgraph = components[np.argmax(number_of_nodes)]
    return subgraph


def closest_point(graph, current_point):
    """
    Compute the closest point in the `graph`
    to the `current_point`.
    """
    all_nodes = np.array(graph.nodes)
    dists = LA.norm(all_nodes - np.array(current_point),axis=1)
    min_ind = np.argmin(dists)
    # need tuple for indexing
    return (all_nodes[min_ind][0],all_nodes[min_ind][1])


def plan_path(start, goal, grid, block_centers, TARGET_ALTITUDE, SAFETY_DISTANCE):
    '''
    the function does the waypoint/path planning
    :param data:
    :param start:
    :param goal:
    :param TARGET_ALTITUDE:
    :param SAFETY_DISTANCE:
    :return:
    '''

    # grid cut at flying altitude
    grid_at_alt = np.array(grid > TARGET_ALTITUDE - SAFETY_DISTANCE, dtype=np.int)

    waypoints = []

    # case 1, check if there is a straight line between  start and goal:
    if ray_tracing_bresham(start, goal, grid_at_alt):
        waypoints = [[start[0], start[1], TARGET_ALTITUDE], [goal[0], goal[1], TARGET_ALTITUDE]]
        return waypoints

    # case 2, create graph for path generattion
    graph = create_graph(block_centers, grid_at_alt)

    start_np = closest_point(graph, start)
    goal_np = closest_point(graph, goal)

    # get straight line and fly by elevate the vehicle to height which no thing will block
    path1 = bresham(start, start_np)
    index = np.transpose(np.array(path1))
    max_altitude = int(np.max(grid[index[0], index[1]]) + SAFETY_DISTANCE)
    if max_altitude < TARGET_ALTITUDE:
        max_altitude = TARGET_ALTITUDE

    waypoints.append([start[0], start[1], max_altitude])
    waypoints.append([start_np[0], start_np[1], max_altitude])
    waypoints.append([start_np[0], start_np[1], TARGET_ALTITUDE])

    path2, _ = a_star_graph(graph, heuristic, start_np, goal_np)
    waypoints = waypoints + [[p[0], p[1], TARGET_ALTITUDE] for p in path2]

    # get straight line and fly by elevate the vehicle to height which no thing will block
    path3 = bresham(goal_np, goal)
    index = np.transpose(np.array(path3))
    max_altitude = int(np.max(grid[index[0], index[1]]) + SAFETY_DISTANCE)
    if max_altitude < TARGET_ALTITUDE:
        max_altitude = TARGET_ALTITUDE

    waypoints.append([goal_np[0], goal_np[1], max_altitude])
    waypoints.append([goal[0], goal[1], max_altitude])
    waypoints.append([goal[0],goal[1],int(grid[goal])])

    pruned_waypoints = prune_path_collinear(waypoints)
    print("Waypoints number before and after Colinear check:", len(waypoints), len(pruned_waypoints))

    return pruned_waypoints


def calc_heading(waypoints):
    '''

    :param waypoints:
    :return: waypoints with heading adjusted
    '''
    waypoints_heading=[]
    wp1 = waypoints[0]
    waypoints_heading.append([wp1[0], wp1[1], wp1[2], 0.0])
    for i in range(1,len(waypoints)):
        wp2 = waypoints[i]
        heading = np.arctan2((wp2[1]-wp1[1]), (wp2[0]-wp1[0]))
        waypoints_heading.append([wp2[0],wp2[1],wp2[2],heading])
        wp1 = wp2

    return waypoints_heading



