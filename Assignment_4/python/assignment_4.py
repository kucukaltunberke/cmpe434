import time
import mujoco 
import mujoco.viewer
import numpy as np
from Controller import Controller
from Circuit import Circuit

# Helper construsts for the viewer for pause/unpause functionality.
paused=False

# Pressing SPACE key toggles the paused state.
def mujoco_viewer_callback(keycode):
    global paused
    if keycode == ord(' '):  # Use ord(' ') for space key comparison
        paused = not paused

def main():
    # Initialize map & simulation
    env = Circuit(model_path="../models/mushr_car/model.xml")
    d = env.d
    m = env.m
    
    # Initialize controller
    kp, kd, ki, dt = 0.3, 0.1, 0.0005, 0.1
    controller = Controller(kp, kd, ki, dt)

    # Launch viewer
    with mujoco.viewer.launch_passive(m, d, key_callback=mujoco_viewer_callback) as viewer:
        throttle = d.actuator("throttle_velocity")
        steering = d.actuator("steering")
        throttle.ctrl=4


        start_time = time.time()
        path_idx = 0

        while viewer.is_running():
            loop_start = time.time()
            if not paused:
                # Current and target positions
                current_pos = d.qpos[:2]
                target_pos = env.reference_path[path_idx]

                w, xq, yq, zq = d.qpos[3:7]

                # 1) signed yaw
                curr_yaw = np.degrees(np.arctan2(2*(w*zq + xq*yq),1 - 2*(yq*yq + zq*zq)))                
                delta_x=target_pos[0]-current_pos[0]
                delta_y=target_pos[1]-current_pos[1]
                target_yaw=np.degrees(np.arctan2(delta_y, delta_x))

                
                steering_correction=controller.update(target_yaw,curr_yaw)
                steering.ctrl=np.clip(steering_correction,-4,4)

                
                print(steering_correction)
                
                # Advance to next waypoint if close
                error = np.linalg.norm([delta_x,delta_y])
                if error < .5:
                    path_idx = (path_idx + 1) % len(env.reference_path)

            # Step simulation & sync viewer
            mujoco.mj_step(m, d)
            viewer.sync()

            # Maintain real-time pacing
            sleep_time = m.opt.timestep - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
