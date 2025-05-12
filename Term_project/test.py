"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı

"""

import math
from enum import Enum
import time
import mujoco
import mujoco.viewer
import random
import numpy as np
import scipy as sp

import cmpe434_dungeon as dungeon
import matplotlib.pyplot as plt
from a_star import AStarPlanner
from dynamic_window_approach import dwa_control, Config as DWAConfig
import matplotlib.pyplot as plt
import numpy as np

show_animation = True

# Helper construsts for the viewer for pause/unpause functionality.
paused = False

# Pressing SPACE key toggles the paused state.
def mujoco_viewer_callback(keycode):
    global paused
    if keycode == ord(' '):  # Use ord(' ') for space key comparison
        paused = not paused

def densify(obstacleList, spacing=0.5):
    dense = []
    for (x,y) in obstacleList:
            # look at your 4‐neighborhood walls and interpolate…
            # simplest: jitter a bit of points inside the square
        for dx in np.arange(-0.2, 0.2, spacing):
            for dy in np.arange(-0.2, 0.2, spacing):
                dense.append((x+dx, y+dy))
    return dense


def get_state(d):
    x, y = d.qpos[0], d.qpos[1]
    w, xq, yq, zq = d.qpos[3:7]
    yaw = np.arctan2(2*(w*zq + xq*yq), 1 - 2*(yq*yq + zq*zq))
    
    # Get velocity in robot frame
    vx, vy, vz = d.qvel[0], d.qvel[1], d.qvel[2]
    v = vx*np.cos(yaw) + vy*np.sin(yaw)  # Forward speed
    
    # Angular velocity (yaw rate)
    omega = d.qvel[5]  # Assuming rotational velocity is in qvel[5]
    
    return np.array([x, y, yaw, v, omega])


def dwa_control(x, config, goal, ob, stuck_count):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory ,stuck_count= calc_control_and_trajectory(x, dw, config, goal, ob, stuck_count)

    return u, trajectory,stuck_count


def A_Star_path_finder(obstacleList,start_pos,final_pos,grid_resolution=0.5,robot_radius=0.5):
    ox , oy = zip(*obstacleList)
    sx , sy = start_pos[0] , start_pos[1]
    gx , gy = final_pos[0] , final_pos[1]

    #Uncomment to see the walls on plot
    plt.plot(ox, oy, ".k")
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")

    a_star = AStarPlanner(ox, oy, grid_resolution, robot_radius)
    rx, ry = a_star.planning(sx, sy, gx, gy)

    rx.reverse()
    ry.reverse()

    #Uncomment to see the A* algorithm on plot
    plt.plot(rx, ry, "-r")
    plt.pause(0.001)
    plt.show()    

    return rx , ry




class RobotType(Enum):
    circle = 0
    rectangle = 1


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
        self.robot_type = RobotType.circle

        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        self.robot_radius = 1.0  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 0.5  # [m] for collision check
        self.robot_length = 1.2  # [m] for collision check
        # obstacles [x(m) y(m), ....]

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


def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


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


def main(robot_type=RobotType.circle):


    # Uncomment to start with an empty model
    # scene_spec = mujoco.MjSpec() 

    # Load existing XML models
    scene_spec = mujoco.MjSpec.from_file("scenes/empty_floor.xml")

    tiles, rooms, connections = dungeon.generate(3, 2, 8)
    for index, r in enumerate(rooms):
        (xmin, ymin, xmax, ymax) = dungeon.find_room_corners(r)
        scene_spec.worldbody.add_geom(name='R{}'.format(index), type=mujoco.mjtGeom.mjGEOM_PLANE, size=[(xmax-xmin)+1, (ymax-ymin)+1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[(xmin+xmax), (ymin+ymax), 0])

    obstacleList=[]

    for pos, tile in tiles.items():
        if tile == "#":
            scene_spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[1, 1, 0.1], rgba=[0.8, 0.6, 0.4, 1],  pos=[pos[0]*2, pos[1]*2, 0])
            
            #To plot the map the wall coordinates are listed.
            obstacleList.append(( pos[0], pos[1]))

    start_pos = random.choice([key for key in tiles.keys() if tiles[key] == "."])
    final_pos = random.choice([key for key in tiles.keys() if tiles[key] == "." and key != start_pos])

    scene_spec.worldbody.add_site(name='start', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.5, 0.01], rgba=[0, 0, 1, 1],  pos=[start_pos[0]*2, start_pos[1]*2, 0])
    scene_spec.worldbody.add_site(name='finish', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.5, 0.01], rgba=[1, 0, 0, 1],  pos=[final_pos[0]*2, final_pos[1]*2, 0])


    robot_spec = mujoco.MjSpec.from_file("models/mushr_car/model.xml")

    # Add robots to the scene:
    # - There must be a frame or site in the scene model to attach the robot to.
    # - A prefix is required if we add multiple robots using the same model.
    scene_spec.attach(robot_spec, frame="world", prefix="robot-")
    scene_spec.body("robot-buddy").pos[0] = start_pos[0] * 2
    scene_spec.body("robot-buddy").pos[1] = start_pos[1] * 2

    # Randomize initial orientation
    yaw = np.random.uniform(-np.pi, np.pi)
    euler = np.array([0.0, 0.0, yaw], dtype=np.float64)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, euler, 'xyz')
    scene_spec.body("robot-buddy").quat[:] = quat

    # Add obstacles to the scene
    for i, room in enumerate(rooms):
        obs_pos = random.choice([tile for tile in room if tile != start_pos and tile != final_pos])
        scene_spec.worldbody.add_geom(
            name='Z{}'.format(i), 
            type=mujoco.mjtGeom.mjGEOM_CYLINDER, 
            size=[0.2, 0.05, 0.1], 
            rgba=[0.8, 0.0, 0.1, 1],  
            pos=[obs_pos[0]*2, obs_pos[1]*2, 0.08]
        )

    # Initalize our simulation
    # Roughly, m keeps static (model) information, and d keeps dynamic (state) information. 
    m = scene_spec.compile()
    d = mujoco.MjData(m)


    obstacles = [m.geom(i).id for i in range(m.ngeom) if m.geom(i).name.startswith("Z")]
    uniform_direction_dist = sp.stats.uniform_direction(2)
    obstacle_direction = [[x, y, 0] for x,y in uniform_direction_dist.rvs(len(obstacles))]
    unused = np.zeros(1, dtype=np.int32)



      # These actuator names are defined in the model XML file for the robot.
      # Prefixes distinguish from other actuators from the same model.
    velocity = d.actuator("robot-throttle_velocity")
    steering = d.actuator("robot-steering")


    raw = obstacleList
    dense = densify(raw, spacing=0.3)


    rx , ry = A_Star_path_finder(dense,start_pos,final_pos)
    target_x_list = [2 * x for x in rx]
    target_y_list = [2 * y for y in ry]
    max_path_id = len(target_x_list)-1
    path_id=0
      # Close the viewer automatically after 30 wall-clock-seconds.
    start = time.time()
              # prepare DWA configuration using default constructor and override defaults
    dwa_config = DWAConfig()
    # tune parameters to match your MuJoCo model
    dwa_config.max_speed = 1.5
    dwa_config.min_speed = -0.4
    dwa_config.max_yaw_rate = 240 * np.pi / 190
    dwa_config.max_accel = 2
    dwa_config.max_delta_yaw_rate = 12 * np.pi 
    dwa_config.v_resolution = 0.1
    dwa_config.yaw_rate_resolution = 6 * np.pi / 90
    dwa_config.dt = 0.1
    dwa_config.predict_time = 3
    dwa_config.to_goal_cost_gain = 2
    dwa_config.speed_cost_gain = .75
    dwa_config.obstacle_cost_gain = 3
    dwa_config.robot_radius = 0.2
    dwa_config.robot_stuck_flag_cons = 0  # constant to prevent robot stucked
    dwa_config.stuckstep =0

    


    stuck_count=0            
    d.qvel[0] = .4 * np.cos(yaw)
    d.qvel[1] = .4 * np.sin(yaw)

    while True:
# initial state [x(m), y(m), yaw(rad), v(m/s), omega(rad/s)]
        state=get_state(d)
        wall_xy = np.array([[x*2, y*2] for x,y in dense])
        dyn_xy  = np.array([[m.geom_pos[g][0], m.geom_pos[g][1]] for g in obstacles])
        ob_xy = np.vstack([wall_xy, dyn_xy])

        path_id_selected = min(path_id + 1, max_path_id)

        target_selected=[target_x_list[path_id_selected],target_y_list[path_id_selected]]

        curr_pos = state[:2]     

        dists       = np.linalg.norm(ob_xy - curr_pos, axis=1)
        ob_in_range = ob_xy[dists <= 6]

        u, _traj ,stuck_count= dwa_control(state, dwa_config,target_selected , ob_in_range, stuck_count)
        v_cmd, omega_cmd = u

            # current_yaw = state[4]
            # steering_correction=pid_controller.update(omega_cmd,current_yaw)

        print(stuck_count)
        print(v_cmd,omega_cmd)
            # MuJoCo’s internal servo P‑controllers handle the low‑level tracking
        velocity.ctrl = v_cmd
        steering.ctrl = np.clip(omega_cmd,-4,4)

        if np.hypot(state[0] - target_selected[0], state[1] - target_selected[1]) < 4:
            if path_id < len(target_x_list) - 1:
                path_id += 1
            elif path_id == max_path_id and np.hypot(state[0] - target_selected[0], state[1] - target_selected[1]) < .3:
                velocity.ctrl = 0
            else:
                pass

            # Update obstables (bouncing movement)
        for i, x in enumerate(obstacles):
            dx = obstacle_direction[i][0]
            dy = obstacle_direction[i][1]

            px = m.geom_pos[x][0]
            py = m.geom_pos[x][1]
            pz = 0.02
            nearest_dist = mujoco.mj_ray(m, d, [px, py, pz], obstacle_direction[i], None, 1, -1, unused)

            if nearest_dist >= 0 and nearest_dist < 0.4:
                obstacle_direction[i][0] = -dy
                obstacle_direction[i][1] = dx

            m.geom_pos[x][0] = m.geom_pos[x][0]+dx*0.001
            m.geom_pos[x][1] = m.geom_pos[x][1]+dy*0.001

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(-_traj[:, 0], _traj[:, 1], "-g")
            plt.plot(state[0], state[1], "xr")
            plt.plot(target_selected[0], target_selected[1], "xb")
            plt.plot(ob_xy[:, 0], ob_xy[:, 1], "ok")
            plot_robot(state[0], state[1], state[2], config)
            plot_arrow(state[0], state[1], state[2])
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)


    print("Done")
    if show_animation:
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    main(robot_type=RobotType.rectangle)
    # main(robot_type=RobotType.circle)
