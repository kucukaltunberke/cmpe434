import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

# Helper construsts for the viewer for pause/unpause functionality.
paused = False

# Pressing SPACE key toggles the paused state.
def mujoco_viewer_callback(keycode):
    global paused
    if keycode == ord(' '):  # Use ord(' ') for space key comparison
        paused = not paused

def construct_walls(scene_spec, wall_height, wall_thickness, center, length, angle):
      x_coor = center[0]
      y_coor = center[1]
      scene_spec.worldbody.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=[x_coor, y_coor, wall_height / 2],
      size=[length, wall_thickness, wall_height],
      euler=[0, 0, angle],
      rgba=[0,.9,.5,1]
    )   

def main():

    # Uncomment to start with an empty model
    scene_spec = mujoco.MjSpec() 
    scene_spec.modelname = "circuit"
    

    wall_height = 0.5
    wall_thickness = 0.1


    # Add a ground plane
    scene_spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, 0],
        size=[50, 50, 0.1],
    )

    construct_walls(scene_spec, wall_height, wall_thickness, [0,1],10,0)
    construct_walls(scene_spec, wall_height, wall_thickness, [0,-1],12,0)

    construct_walls(scene_spec, wall_height, wall_thickness, [12,6],7,90)
    construct_walls(scene_spec, wall_height, wall_thickness, [10,6],5,90)

    construct_walls(scene_spec, wall_height, wall_thickness, [6,11],4,0)
    construct_walls(scene_spec, wall_height, wall_thickness, [6,13],6,0)

    construct_walls(scene_spec, wall_height, wall_thickness, [2,8],3,90)
    construct_walls(scene_spec, wall_height, wall_thickness, [0,10],3,90)
    
    construct_walls(scene_spec, wall_height, wall_thickness, [-4,5],6,0)
    construct_walls(scene_spec, wall_height, wall_thickness, [-6,7],6,0)

    construct_walls(scene_spec, wall_height, wall_thickness, [-10,3],2,90)
    construct_walls(scene_spec, wall_height, wall_thickness, [-12,3],4,90)

    
    world_frame = scene_spec.worldbody.add_frame(name="world", pos=[0, 0, 0])

    # Load existing XML models
    #scene_spec = mujoco.MjSpec.from_file("scenes/empty_floor.xml")
    robot_spec = mujoco.MjSpec.from_file("models/mushr_car/model.xml")



    # Add robots to the scene:
    # - There must be a frame or site in the scene model to attach the robot to.
    # - A prefix is required if we add multiple robots using the same model.
    scene_spec.attach(robot_spec, frame="world", prefix="robot-")

    # Initalize our simulation
    # Roughly, m keeps static (model) information, and d keeps dynamic (state) information. 
    m = scene_spec.compile()
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d, key_callback=mujoco_viewer_callback) as viewer:

      # These actuator names are defined in the model XML file for the robot.
      # Prefixes distinguish from other actuators from the same model.
      velocity = d.actuator("robot-throttle_velocity")
      steering = d.actuator("robot-steering")

      positions=[]
      velocities=[]
      accelerations=[]
      start_time = time.time()  # Get the start time of the simulation

      while viewer.is_running():
          step_start = time.time()
          elapsed_time = step_start - start_time  # Calculate the elapsed time since the start of the simulation

          if not paused:
              # Apply velocity and steering control based on elapsed time
              if 0<d.qpos[0] < 10 and d.qpos[1] <1:
                  velocity.ctrl = 3.0  # Set velocity
                  steering.ctrl = 0.0  # Set steering
              elif d.qpos[0] > 10 and d.qpos[1] <1:
                  velocity.ctrl = 1.0  # Set velocity
                  steering.ctrl = 4.0  # Set steering
              elif d.qpos[0] > 10 and d.qpos[1] <10.5:
                  velocity.ctrl = 3.0  # Set velocity
                  steering.ctrl = 0.0  # Set steering
              elif d.qpos[0] > 10 and d.qpos[1] >10.5:
                  velocity.ctrl = 1.0  # Set velocity
                  steering.ctrl = 4.0  # Set steering
              elif 2 < d.qpos[0] < 10 and d.qpos[1] > 10.5:
                  velocity.ctrl = 3.0  # Set velocity
                  steering.ctrl = 0.0  # Set steering
              elif d.qpos[0] < 2 and d.qpos[1] > 10.5:
                  velocity.ctrl = 1.0  # Set velocity
                  steering.ctrl = 4.0  # Set steering              
              elif d.qpos[0] < 2 and 7 <d.qpos[1] < 10.5:
                  velocity.ctrl = 3.0  # Set velocity
                  steering.ctrl = 0.0  # Set steering
              elif d.qpos[0] > 0 and d.qpos[1] < 7:
                  velocity.ctrl = 1.0  # Set velocity
                  steering.ctrl = -4  # Set steering        
              elif -10<d.qpos[0] and d.qpos[1] < 7:
                  velocity.ctrl = 3.0  # Set velocity
                  steering.ctrl = 0.0  # Set steering
              elif d.qpos[0] < -10  and d.qpos[1]>5:
                  velocity.ctrl = 1.0  # Set velocity
                  steering.ctrl = 3.8  # Set steering
              elif d.qpos[0]<-10 and d.qpos[1] > 1:
                  velocity.ctrl = 3.0  # Set velocity
                  steering.ctrl = 0.0  # Set steering
              elif d.qpos[0] < -10  and d.qpos[1]<1:
                  velocity.ctrl = 1.0  # Set velocity
                  steering.ctrl = 3.8  # Set steering
              elif d.qpos[0]>-10 and d.qpos[1] < 1:
                  velocity.ctrl = 3.0  # Set velocity
                  steering.ctrl = 0.0  # Set steering

              # Save positions, velocities, and accelerations
              positions.append(d.qpos.copy())  # Position (x, y, z)
              velocities.append(d.qvel.copy())  # Velocity (vx, vy, vz)
              accelerations.append(d.qacc.copy())  # Acceleration (ax, ay, az)

          # Step the simulation
          mujoco.mj_step(m, d)

          # Synchronize viewer with simulation
          viewer.sync()


        # Rudimentary time keeping, will drift relative to wall clock.
          time_until_next_step = m.opt.timestep - (time.time() - step_start)
          if time_until_next_step > 0:
           time.sleep(time_until_next_step)

    positions = np.array(positions)
    velocities = np.array(velocities)
    accelerations = np.array(accelerations)


    # Plot the car's trajectory (x vs. y positions)
    plt.subplot(3, 1, 1)
    plt.plot(positions[:, 0], positions[:, 1], label="Car Trajectory")
    plt.title("Car Trajectory")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()

    # Plot velocity components over time
    plt.subplot(3, 1, 2)
    plt.plot(velocities[:, 0], label="X Velocity")
    plt.plot(velocities[:, 1], label="Y Velocity")
    plt.title("Velocity Components Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity (m/s)")
    plt.legend()

    # Plot acceleration components over time
    plt.subplot(3, 1, 3)
    plt.plot(accelerations[:, 0], label="X Acceleration")
    plt.plot(accelerations[:, 1], label="Y Acceleration")
    plt.title("Acceleration Components Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

