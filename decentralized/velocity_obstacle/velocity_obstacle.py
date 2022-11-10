"""
Collision avoidance using Velocity-obstacle method

author: Ashwin Bose (atb033@github.com)
"""
import sys
sys.path.append('D:\multi_agent_path_planning\decentralized')
from utils.control import compute_desired_velocity
from utils.create_obstacles import create_obstacles
from utils.multi_robot_plot import plot_robot_and_obstacles
import numpy as np


SIM_TIME = 5.
TIMESTEP = 0.1
NUMBER_OF_TIMESTEPS = int(SIM_TIME/TIMESTEP)
ROBOT_RADIUS = 0.5
VMAX = 2
VMIN = 0.2


def simulate(filename):
    obstacles = create_obstacles(SIM_TIME, NUMBER_OF_TIMESTEPS)

    start = np.array([5, 0, 0, 0])
    goal = np.array([5, 10, 0, 0])

    robot_state = start
    robot_state_history = np.empty((4, NUMBER_OF_TIMESTEPS))
    for i in range(NUMBER_OF_TIMESTEPS):
        v_desired = compute_desired_velocity(robot_state, goal, ROBOT_RADIUS, VMAX)
        control_vel = compute_velocity(robot_state, obstacles[:, i, :], v_desired)
        robot_state = update_state(robot_state, control_vel)
        robot_state_history[:4, i] = robot_state

    plot_robot_and_obstacles(robot_state_history, obstacles, ROBOT_RADIUS, NUMBER_OF_TIMESTEPS, SIM_TIME, filename)


def compute_velocity(robot, obstacles, v_desired):
    pA = robot[:2]  # take the first 2 element of the array
    vA = robot[-2:]  # take the last 2 element of the array
    # Compute the constraints
    # for each velocity obstacles
    number_of_obstacles = np.shape(obstacles)[1]
    Amat = np.empty((number_of_obstacles * 2, 2))
    bvec = np.empty((number_of_obstacles * 2))
    for i in range(number_of_obstacles):  # compute velocity obstacles for each obstacles/robots
        obstacle = obstacles[:, i]  # "obstacles" each column is represented as a robot, this is to get the first column
        pB = obstacle[:2]  # first two element of the column
        vB = obstacle[2:]  # last two element of the column
        dispBA = pA - pB
        distBA = np.linalg.norm(dispBA)
        thetaBA = np.arctan2(dispBA[1], dispBA[0])
        if 2.2 * ROBOT_RADIUS > distBA:
            distBA = 2.2*ROBOT_RADIUS
        phi_obst = np.arcsin(2.2*ROBOT_RADIUS/distBA)
        phi_left = thetaBA + phi_obst
        phi_right = thetaBA - phi_obst

        # VO
        translation = vB
        Atemp, btemp = create_constraints(translation, phi_left, "left")  # the line extend from left
        Amat[i*2, :] = Atemp
        bvec[i*2] = btemp
        Atemp, btemp = create_constraints(translation, phi_right, "right")  # the line extend from right
        Amat[i*2 + 1, :] = Atemp
        bvec[i*2 + 1] = btemp

    # Create search-space, combination of theta with speed will form the combined search space
    th = np.linspace(0, 2*np.pi, 20)  # create search space of  theta
    vel = np.linspace(0, VMAX, 5)  # create search space of Speed

    vv, thth = np.meshgrid(vel, th)

    vx_sample = (vv * np.cos(thth)).flatten()
    vy_sample = (vv * np.sin(thth)).flatten()

    v_sample = np.stack((vx_sample, vy_sample))  # sample 100 velocities

    v_satisfying_constraints = check_constraints(v_sample, Amat, bvec)

    # Objective function
    size = np.shape(v_satisfying_constraints)[1]
    diffs = v_satisfying_constraints - ((v_desired).reshape(2, 1) @ np.ones(size).reshape(1, size))
    norm = np.linalg.norm(diffs, axis=0)
    min_index = np.where(norm == np.amin(norm))[0][0]
    cmd_vel = (v_satisfying_constraints[:, min_index])  # identify the velocity (from the admissiable velocity pool) that has the smallest difference from the reference velocity at each time-step.

    return cmd_vel


def check_constraints(v_sample, Amat, bvec):
    length = np.shape(bvec)[0]

    for i in range(int(length/2)):
        v_sample = check_inside(v_sample, Amat[2*i:2*i+2, :], bvec[2*i:2*i+2])

    return v_sample


def check_inside(v, Amat, bvec):
    v_out = []
    for i in range(np.shape(v)[1]):
        if not ((Amat @ v[:, i] < bvec).all()):
            v_out.append(v[:, i])
    return np.array(v_out).T


def create_constraints(translation, angle, side):
    # create line
    origin = np.array([0, 0, 1])
    point = np.array([np.cos(angle), np.sin(angle)])  # unit vector in both x and y direction
    # when use {0,0,1} cross product with a unit vector, the result will produce {-np.sin(angle), np.cos(angle), 0}
    line = np.cross(origin, point)  # for cross product, if the vector do not have enough dimensions, the vector with lessor dimension will be automatically filled with an zero.
    line = translate_line(line, translation)  #

    if side == "left":
        line *= -1  # line = line + -1

    A = line[:2]
    b = -line[2]

    return A, b


def translate_line(line, translation):
    matrix = np.eye(3)
    matrix[2, :2] = -translation[:2]  # [:2] is to get the first two element, and at least get first two element, if there are only two element, we just get the two element
    # matrix[2,:] is to get the third row of the array "matrix", and :2 mean 2 columns
    # @ sign is the matrix multiplication
    return matrix @ line


def update_state(x, v):
    new_state = np.empty((4))
    new_state[:2] = x[:2] + v * TIMESTEP
    new_state[-2:] = v
    return new_state
