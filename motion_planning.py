import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np
import re as re
from time import gmtime, strftime

from planning_utils import create_grid, plan_path, calc_heading
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local,local_to_global

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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
                print("Local_position, target_position", self.local_position * -1, self.target_position)
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            # add deadband control
            deadband = np.clip(np.linalg.norm(self.local_velocity)/1.0,0.1,5)
            if abs(np.linalg.norm(self.target_position[0:2] - self.local_position[0:2])) < deadband and \
                abs(self.target_position[2] + self.local_position[2]) < 0.2:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.landing_height< 0.1:
                if self.local_position[2]-self.landing_height < 0.01:
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
        TARGET_ALTITUDE = 10
        SAFETY_DISTANCE = 10

        self.target_position[2] = TARGET_ALTITUDE

        # read lat0, lon0 from colliders into floating point values
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
        
        # set home position to (lon0, lat0, 0)
        self.set_home_position(lon0,lat0,0.0)

        # retrieve current global position
        # global_position = self.global_position
 
        # convert to current local position using global_to_local()
        # self.local_position  = global_to_local(global_position,self.global_home)
        
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        grid_w_safty, grid, north_offset, east_offset = create_grid(data, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))

        # convert start position to current position rather than map center

        (north_size, east_size) = grid_w_safty.shape
        grid_start = (int(self.local_position[0]) - north_offset,int(self.local_position[1]) - east_offset)
        # Set goal as some arbitrary position on the grid
        # random pick some goal
        (north_size, east_size) = grid_w_safty.shape

        # comment this line to choose a goal manually
        # the drone is supposed to fly to anywhere on the map, as it can fly.
        grid_goal = (np.random.choice(north_size-1,1)[0],np.random.choice(east_size-1,1)[0])
        # manual goal
        #grid_goal = (210,100)

        # TODO: adapt to set goal as latitude / longitude position and convert
        # Not necessary to do so maybe just for information.
        goal_global = local_to_global([grid_goal[0]+north_offset,grid_goal[1]+east_offset,grid_w_safty[grid_goal]+TARGET_ALTITUDE-SAFETY_DISTANCE], self.global_home)
        print("Goal global position: ",goal_global)


        # or move to a different search space such as a graph (not done here)
        start_3d = (grid_start[0],grid_start[1],-self.local_position[2])
        goal_3d = (grid_goal[0],grid_goal[1],grid[grid_goal])
        print('Start and Goal position on grid: ', grid_start, grid_goal)

        # block center at flying altitude
        grid_block_centers = data[np.where(data[:, 2] * 2 + SAFETY_DISTANCE > TARGET_ALTITUDE)][:, :2]  - [north_offset,east_offset]

        # waypoint in grid coordinate
        grid_waypoints = plan_path(start_3d, goal_3d, grid_w_safty, grid_block_centers, TARGET_ALTITUDE, SAFETY_DISTANCE)

        # Convert grid_waypoints to NED local coordination
        waypoints = np.array(grid_waypoints) + np.array([north_offset,east_offset,0])
        # Add heading in waypoint
        waypoints = calc_heading(waypoints)

        # set the landing height, can be on building
        self.landing_height = waypoints[-1][2]
        print("Landing height:", self.landing_height)
        # Set self.waypoints
        # don't know why send_waypoints will fail if send waypoints directly
        # through this convert it works.
        self.waypoints = [ [int(wp[0]),int(wp[1]),int(wp[2]),wp[3]] for wp in waypoints]
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

        # just re assign the original waypoints
        self.waypoints = waypoints

        # to output a picture for navigation
        plt.figure(figsize=(12, 12))
        plt.imshow(grid > TARGET_ALTITUDE - SAFETY_DISTANCE, origin='lower')
        plt.xlabel('EAST')
        plt.ylabel('NORTH')
        pp = np.transpose(np.array(grid_waypoints))
        plt.plot(pp[1], pp[0], c='g')
        plt.plot(grid_start[1], grid_start[0], marker='o', c='b')
        plt.plot(grid_goal[1],  grid_goal[0], '*')
        plt.savefig("plot_{}.png".format(strftime("%Y_%m_%d_%H_%M_%S", gmtime())))



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
