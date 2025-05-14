import numpy as np
import matplotlib.pyplot as plt
from Circuit import Circuit
from probabilistic_road_map.probabilistic_road_map import prm_planning
from AStar.a_star import AStarPlanner
from RRT.rrt import RRT
from RRTDubins.rrt_dubins import RRTDubins
from RRTStar.rrt_star import RRTStar
from RRTStarDubins.rrt_star_dubins import RRTStarDubins

show_animation = True

def circuit2dots(circuit,resolution): 
    ox, oy=[],[]
    for center, length , angle in circuit.wall_info:
        c_x , c_y = center[0] , center[1]
        angle_rad=np.radians(angle)
        number_of_dots=int(np.ceil(length/resolution))
        length *= 2 #length corrrection
        for i in range(number_of_dots+1):
            offset = -length / 2 + i * (length / number_of_dots)
            x = c_x + offset * np.cos(angle_rad)
            y = c_y + offset * np.sin(angle_rad)
            ox.append(x)
            oy.append(y)

    for i in range(6):
        ox.append(0)
        oy.append(-1+float(i)/3)        
    return ox, oy

def main():
    env = Circuit(model_path="../models/mushr_car/model.xml")
    grid_resolution = 0.2 
    robot_radius=0.3
    sx , sy = (3,0)
    gx , gy = (-3,0)

    ox, oy = circuit2dots(env, grid_resolution)

    algorithm_type=int(input("Which algorithm would you like to run? 1 for ProbabilisticRoadMap, 2 for RRT, 3 for RRTDubins, 4 for RRTStar, 5 for RRTStarDubins, 6 for A* "))

    if show_animation:  # pragma: no cover
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")
    
    if algorithm_type == 1:
        rx, ry = prm_planning(sx, sy, gx, gy, ox, oy, robot_radius, rng=None)

        if show_animation:  # pragma: no cover
            plt.plot(rx, ry, "-r")
            plt.pause(0.001)
            plt.show()

    elif algorithm_type == 2:
        obstacleList = []
        for i in range(len(ox)): obstacleList.append(( ox[i], oy[i] , 0.1))

        rrt = RRT(
        start=[sx, sy],
        goal=[gx, gy],
        rand_area=[-8, 23],
        obstacle_list=obstacleList,
        play_area=[-13, 13, -3, 14],
        robot_radius=robot_radius)

        path = rrt.planning(animation=show_animation)
        
        if path is None:
            print("Cannot find path")
        else:
            print("found path!!")

            # Draw final path
            if show_animation:
                plt.axis([-3, 3, -3, 3])
                rrt.draw_graph()
                plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
                plt.grid(True)
                plt.pause(0.01)  # Need for Mac
                plt.show()

    elif algorithm_type == 3:
        obstacleList = []
        for i in range(len(ox)): obstacleList.append(( ox[i], oy[i] , 0.1))

        # Set Initial parameters
        start = [float(sx),float(sy) , np.deg2rad(0.0)]
        goal = [float(gx), float(gy), np.deg2rad(0.0)]

        rrt_dubins = RRTDubins(start, goal, obstacleList, [-15.0, 15.0])
        path = rrt_dubins.planning(animation=show_animation)

        # Draw final path
        if show_animation:  # pragma: no cover
            rrt_dubins.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.001)
            plt.show()

    elif algorithm_type == 4:
        obstacleList = []
        for i in range(len(ox)): obstacleList.append(( ox[i], oy[i] , 0.1))

        # Set Initial parameters
        start = [float(sx),float(sy) , np.deg2rad(0.0)]
        goal = [float(gx), float(gy), np.deg2rad(0.0)]

        rrt_star = RRTStar(
            start=start,
            goal=goal,
            rand_area=[-13, 20],
            obstacle_list=obstacleList,
            expand_dis=3,
            robot_radius=0.4)
        path = rrt_star.planning(animation=show_animation)

        if path is None:
            print("Cannot find path")
        else:
            print("found path!!")

            # Draw final path
            if show_animation:
                rrt_star.draw_graph()
                plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
                plt.grid(True)
                plt.show()

    elif algorithm_type == 5:
        obstacleList = []
        for i in range(len(ox)): obstacleList.append(( ox[i], oy[i] , 0.1))

        # Set Initial parameters
        start = [float(sx),float(sy) , np.deg2rad(0.0)]
        goal = [float(gx), float(gy), np.deg2rad(0.0)]

        rrtstar_dubins = RRTStarDubins(start, goal, rand_area=[-13.0, 15.0], obstacle_list=obstacleList)
        path = rrtstar_dubins.planning(animation=show_animation)

        # Draw final path
        if show_animation:  # pragma: no cover
            rrtstar_dubins.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.001)

            plt.show()

    elif algorithm_type == 6:
        a_star = AStarPlanner(ox, oy, grid_resolution, robot_radius)
        rx, ry = a_star.planning(sx, sy, gx, gy)

        if show_animation:  # pragma: no cover
            plt.plot(rx, ry, "-r")
            plt.pause(0.001)
            plt.show()

    else:
        print("Please run the program again with correct input integers ")

if __name__ == "__main__":
    main()
