import mujoco


class Circuit:
    def __init__(self,model_path, wall_height=0.5, wall_thickness= 0.1):
        self.scene_spec = mujoco.MjSpec()
        self.wall_height=wall_height
        self.wall_thickness = wall_thickness

        #wall_info is contstucted in a way that each array gives information
        #about center, lenght and angle of the wall.
        wall_info = [                   
            ([0, 1], 10, 0), ([0, -1], 12, 0),
            ([12, 6], 7, 90), ([10, 6], 5, 90),
            ([6, 11], 4, 0), ([6, 13], 6, 0),
            ([2, 8], 3, 90), ([0, 10], 3, 90),
            ([-4, 5], 6, 0), ([-6, 7], 6, 0),
            ([-10, 3], 2, 90), ([-12, 3], 4, 90),
        ]

        self.reference_path = [
            [10, 0], [11, 5], [11, 11.5], [7, 12],
            [2, 12], [1, 9], [1, 6], [-3, 6],
            [-10, 6], [-11, 4], [-11, 1], [-7, 0],
        ]       

        self.construct_circuit(wall_info)
        self.add_robot(model_path)

        self.m = self.scene_spec.compile()
        self.d = mujoco.MjData(self.m)

    def construct_circuit(self,wall_info):

        self.scene_spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        pos=[0, 0, 0],
        size=[50, 50, 0.1],
        rgba=[1,1,1,1],
    )
        
        self.scene_spec.worldbody.add_frame(
            name="world", pos=[0, 0, 0]
        )

        for center, length, angle in wall_info:
            self.add_wall(center, length, angle)


    def add_wall(self, center, length, angle):
        x, y = center
        self.scene_spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, y, self.wall_height / 2],
            size=[length, self.wall_thickness, self.wall_height],
            euler=[0, 0, angle],
            rgba=[1, 1, 0.5, 1],)        

    def add_robot(self, model_path):
        robot_spec = mujoco.MjSpec.from_file(model_path)
        self.scene_spec.attach(robot_spec, frame="world")


