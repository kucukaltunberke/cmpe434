import numpy as np

class PIDcontroller:
    def __init__(self, kp, kd, ki, dt):
        self.kp=kp
        self.ki=ki
        self.kd=kd
        self.dt=dt
        self.prev_err=0
        self.error_sum=0

    def update(self,target_pos, curr_pos):
        target_pos = np.array(target_pos)
        curr_pos=np.array(curr_pos)

        #error angle is wrapped into[-180,180]
        err = (target_pos - curr_pos + 180) % 360 - 180

        p=self.kp* err
        
        self.error_sum += err * self.dt
        i=self.error_sum * self.ki

        derivative = (err-self.prev_err)/self.dt
        self.prev_err=err
        d=derivative * self.kd
        

        return p + i + d
