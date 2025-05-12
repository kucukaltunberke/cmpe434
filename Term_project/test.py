"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

"""

import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

show_animation = True


def dwa_control(x, config, goal, ob, stuck_count):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory ,stuck_count= calc_control_and_trajectory(x, dw, config, goal, ob, stuck_count)

    return u, trajectory,stuck_count

class RobotType(Enum):
    circle = 1
    rectangle = 0


class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # robot parameter
        self.max_speed = 1.0  # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s]
        self.max_accel = 0.2  # [m/ss]
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/ss]
        self.v_resolution = 0.01  # [m/s]
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 3.0  # [s]
        self.to_goal_cost_gain = 0.15
        self.speed_cost_gain = 1.0
        self.obstacle_cost_gain = 1.0
        self.robot_stuck_flag_cons = 0.001  # constant to prevent robot stucked
        self.stuckstep = 1000
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        # obstacles [x(m) y(m), ....]
        self.ob = np.array([[-1, -1],
                            [0, 2],
                            [4.0, 2.0],
                            [5.0, 4.0],
                            [5.0, 5.0],
                            [5.0, 6.0],
                            [5.0, 9.0],
                            [8.0, 9.0],
                            [7.0, 9.0],
                            [8.0, 10.0],
                            [9.0, 11.0],
                            [12.0, 13.0],
                            [12.0, 12.0],
                            [15.0, 15.0],
                            [13.0, 13.0]
                            ])

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


config = Config()


def motion(x, u, dt):
    """
    motion model
    """

    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob, stuck_count):
    """
    Calculate the best [v, omega] using Dynamic Window Approach,
    then, if the robot is stuck, blend that v toward a fixed backward speed.
    Returns: best_u = [v_cmd, omega_cmd], best_trajectory, updated stuck_count
    """
    # 1) Unpack state
    curr_x, curr_y, curr_yaw, curr_v, curr_omega = x

    # 2) Helper to wrap into (-π, π]
    def normalize(a):
        return (a + np.pi) % (2*np.pi) - np.pi

    # 3) Compute signed heading error
    bearing_to_goal = np.arctan2(goal[1] - curr_y,
                                 goal[0] - curr_x)
    angle_diff =- normalize(bearing_to_goal - curr_yaw)

    # 4) NORMAL DWA SEARCH
    min_cost     = float("inf")
    best_v       = 0.0
    best_omega   = 0.0
    best_traj    = np.array([x])

    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for omega in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            traj = predict_trajectory(x, v, omega, config)

            c_goal  = config.to_goal_cost_gain   * calc_to_goal_cost(traj, goal)
            c_speed = config.speed_cost_gain     * (config.max_speed - traj[-1,3])
            c_obs   = config.obstacle_cost_gain  * calc_obstacle_cost(traj, ob, config)

            cost = c_goal + c_speed + c_obs

            if cost < min_cost:
                min_cost   = cost
                best_v     = v
                best_omega = omega
                best_traj  = traj

    # 5) Detect entering "stuck" mode
    if stuck_count == 0 and \
       abs(best_v) < config.robot_stuck_flag_cons and \
       abs(curr_v) < config.robot_stuck_flag_cons:
        stuck_count = 1

    # 6) Compute blend factor β ∈ [0,1]
    if stuck_count > 0:
        beta = min(stuck_count / config.stuckstep, 1.0)
    else:
        beta = 0.0

    # 7) Mix forward (best_v) with backward (-0.2)
    backward_speed = -0.3
    v_cmd = (1 - beta) * best_v + beta * backward_speed

    # 8) Mix heading: DWA’s omega vs. a fixed break-out turn
    breakout_omega = np.sign(angle_diff) * 3
    omega_cmd     = (1 - beta) * best_omega + beta * breakout_omega

    # 9) Advance or reset stuck_count
    if stuck_count > 0:
        stuck_count += 1
        if stuck_count >= config.stuckstep:
            stuck_count = 0

    # 10) (Optional) build a one-step trajectory for visualization
    blended_traj = predict_trajectory(x, v_cmd, omega_cmd, config)

    return [v_cmd, omega_cmd], blended_traj, stuck_count


def calc_obstacle_cost(trajectory, ob, config):
    """
    calc obstacle cost inf: collision
    """
    if ob.shape[0] == 0:
        return  0.0
    

    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")
    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    min_r = np.min(r)
    return 1.0 / min_r  # OK



def calc_to_goal_cost(trajectory, goal, alpha=0, beta=1.5):
    # α⋅distance + β⋅heading‐error
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    dist_cost = np.hypot(dx, dy)

    # heading error
    goal_yaw   = math.atan2(dy, dx)
    traj_yaw   = trajectory[-1, 2]
    yaw_error  = abs(math.atan2(
                    math.sin(goal_yaw-traj_yaw),
                    math.cos(goal_yaw-traj_yaw)))
    return alpha * dist_cost + beta * yaw_error



def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    if config.robot_type == RobotType.rectangle:
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")
    elif config.robot_type == RobotType.circle:
        circle = plt.Circle((x, y), config.robot_radius, color="b")
        plt.gcf().gca().add_artist(circle)
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")


def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):
    print(__file__ + " start!!")
    # initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # input [forward speed, yaw_rate]

    config.robot_type = robot_type
    trajectory = np.array(x)
    ob = config.ob
    while True:
        u, predicted_trajectory = dwa_control(x, config, goal, ob)
        x = motion(x, u, config.dt)  # simulate robot
        trajectory = np.vstack((trajectory, x))  # store state history

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(x[0], x[1], "xr")
            plt.plot(goal[0], goal[1], "xb")
            plt.plot(ob[:, 0], ob[:, 1], "ok")
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main(robot_type=RobotType.rectangle)
    # main(robot_type=RobotType.circle)
