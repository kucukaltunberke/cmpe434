import time
import mujoco 
import mujoco.viewer
import numpy as np
import argparse
from controller4 import PIDcontroller
from controller5 import Pure_pursuit
from Circuit import Circuit

# Helper construsts for the viewer for pause/unpause functionality.
paused=False

# Pressing SPACE key toggles the paused state.
def mujoco_viewer_callback(keycode):
    global paused
    if keycode == ord(' '):  # Use ord(' ') for space key comparison
        paused = not paused

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=int, choices=[4, 5], required=True)
    parser.add_argument("--velocity",   type=int, choices=range(0, 11), required=True)

    args = parser.parse_args()

    controller_type = args.controller
    speed         = args.velocity

    # Initialize map & simulation
    env = Circuit(model_path="../models/mushr_car/model.xml")
    d = env.d
    m = env.m
    
    # Initialize controller


    # Launch viewer
    with mujoco.viewer.launch_passive(m, d, key_callback=mujoco_viewer_callback) as viewer:
        throttle = d.actuator("throttle_velocity")
        steering = d.actuator("steering")
        throttle.ctrl=speed


        start_time = time.time()
        path_idx = 0

        kp, kd, ki, dt = 0.3, 0.03, 0.0001, 0.1
        pid_controller = PIDcontroller(kp, kd, ki, dt)

        look_ahead_distance=1                    
        pure_pursuit_controller=Pure_pursuit(look_ahead_distance)

        while viewer.is_running():
            loop_start = time.time()
            if not paused:
                # Current and target positions
                current_pos = d.qpos[:2]
                w, xq, yq, zq = d.qpos[3:7]
                curr_yaw = np.degrees(np.arctan2(2*(w*zq + xq*yq),1 - 2*(yq*yq + zq*zq)))                
                
                if controller_type==4:
                    
                    target_pos = env.reference_path[path_idx]

                    delta_x=target_pos[0]-current_pos[0]
                    delta_y=target_pos[1]-current_pos[1]
                    target_yaw=np.degrees(np.arctan2(delta_y, delta_x))

                    
                    

                    steering_correction=pid_controller.update(target_yaw,curr_yaw)
                    
                    print(steering_correction,"         ", d.qvel[5])
                    steering.ctrl=np.clip(steering_correction,-4,4)
                                        

                    # Advance to next waypoint if close
                    error = np.linalg.norm([delta_x,delta_y])
                    if error < .5:
                        path_idx = (path_idx + 1) % len(env.reference_path)

                elif controller_type ==5:
                    

                    point_behind=env.reference_path[path_idx-1]
                    point_ahead=env.reference_path[path_idx]


                    intersections=pure_pursuit_controller.line_circle_intersection(current_pos,point_ahead,point_behind)
                    if intersections==None:
                        target_pos=point_ahead
                    else:
                        target_pos=pure_pursuit_controller.find_target_point(intersections,point_ahead,point_behind)
                    

                    delta_x=target_pos[0]-current_pos[0]
                    delta_y=target_pos[1]-current_pos[1]
                    target_yaw=np.degrees(np.arctan2(delta_y, delta_x))


                    steering_correction=pid_controller.update(target_yaw,curr_yaw)
                    steering.ctrl=np.clip(steering_correction,-4,4)

                    dist_next_point = np.linalg.norm(np.array(point_ahead) - np.array(current_pos))

                    
                    if dist_next_point <.9:
                        path_idx += 1
                        if path_idx==11:
                            path_idx=0

                # Step simulation & sync viewer
                mujoco.mj_step(m, d)
                viewer.sync()

                # Maintain real-time pacing
                sleep_time = m.opt.timestep - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)


if __name__ == "__main__":
    main()
