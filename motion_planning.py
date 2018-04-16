import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import re as re

from planning_utils import heuristic, create_grid_and_graph,closest_point,a_star_graph, \
    prune_path_ray_tracing, prune_path_collinear,calc_heading, connect_to_goal
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local,local_to_global


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL
        self.landing_height = 0.0

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            # add deadband control
            deadband = np.linalg.norm(self.local_velocity)/1.0
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < deadband:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.landing_height< 0.1:
                if abs(self.local_position[2]-self.landing_height) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        # read the first line of 'colliders.csv'
        first_line = ""
        with open('colliders.csv') as file:
            first_line = file.readline()
        # extract lat0 and lon0 with regular expression
        re_floating = "([+-]?[0-9]*.?[0-9]+)"
        re_expression = "lat0 {}, lon0 {}".format(re_floating,re_floating)
        m = re.search(re_expression, first_line)

        lat0 = 0.
        lon0 = 0.
        if m is not None:
            lat0 = float(m.group(1))
            lon0 = float(m.group(2))
        else:
            print("Error: Read lat0 and lon0")
        
        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(lon0,lat0,0.0)

        # TODO: retrieve current global position
        # global_position = self.global_position
 
        # TODO: convert to current local position using global_to_local()
        # self.local_position  = global_to_local(global_position,self.global_home)
        
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        #grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        grid, graph,north_offset, east_offset = create_grid_and_graph(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # Define starting point on the grid (this is just grid center)
        # grid_start = (-north_offset, -east_offset)
        # TODO: convert start position to current position rather than map center
        (north_size,east_size) = grid.shape
        grid_start = (int(np.clip(self.local_position[0] - north_offset,0,north_size-1)),
                      int(np.clip(self.local_position[1] - east_offset, 0, east_size - 1)))
        # Set goal as some arbitrary position on the grid
        # random pick some goal
        #grid_goal = (np.random.choice(north_size-1,1)[0],np.random.choice(east_size-1,1)[0])
        grid_goal = (210,100)
        # TODO: adapt to set goal as latitude / longitude position and convert
        # Not necessary to do so maybe just for information.
        goal_global = local_to_global([grid_goal[0]+north_offset,grid_goal[1]+east_offset,0.], self.global_home)
        print("Goal global position: ",goal_global)
        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        print('Local Start and Goal: ', grid_start, grid_goal)
        grid_start_np = closest_point(graph,grid_start)
        grid_goal_np = closest_point(graph,grid_goal)
        path, _ = a_star_graph(graph, heuristic, grid_start_np, grid_goal_np)
        # TODO: prune path to minimize number of waypoints
        pruned_path = prune_path_collinear(path)
        pruned_path = prune_path_ray_tracing(pruned_path,grid)
        print("Waypoints before pruning {} and after prunning {}".format(len(path),len(pruned_path)))
        # TODO (if you're feeling ambitious): Try a different approach altogether!

        # Convert path to waypoints
        waypoints = [[ p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in pruned_path]

        # add waypoints from grid_goal_np to grid_goal
        connect_path = connect_to_goal(grid,grid_goal_np,grid_goal)
        for p in connect_path:
            waypoints.append([ p[0] + north_offset, p[1] + east_offset, p[2]+TARGET_ALTITUDE, 0])

        waypoints = calc_heading(waypoints)

        self.landing_height = waypoints[-1][2] - SAFETY_DISTANCE
        print("Landing height:", self.landing_height)

        waypoints = [ [int(wp[0]),int(wp[1]),int(wp[2]),wp[3]] for wp in waypoints]
        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
