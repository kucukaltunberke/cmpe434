import numpy as np

class Pure_pursuit:
    def __init__(self,look_ahead_distance):
        self.look_ahead_distance=look_ahead_distance


    def line_circle_intersection(self,currentPos, point_ahead, point_behind):

        currentX , currentY= currentPos

        #Naming the point behind as 1 and point ahead as 2 for convenience.
        x1 , y1= point_behind  
        x2 , y2= point_ahead

        x1_diff = x1 - currentX
        y1_diff = y1 - currentY
        x2_diff = x2 - currentX
        y2_diff = y2 - currentY  
        
        dx = x2_diff - x1_diff
        dy = y2_diff - y1_diff
        dr = np.sqrt (dx**2 + dy**2)
        D = x1_diff*y2_diff - x2_diff*y1_diff
        discriminant = (self.look_ahead_distance**2) * (dr**2) - D**2  

        # if discriminant is >= 0, there exist solutions
        if discriminant >= 0:
        
            # calculate the solutions

            sign_dy = 1 if dy >= 0 else -1
            sol_x1 = (D * dy + sign_dy * dx * np.sqrt(discriminant)) / dr**2
            sol_x2 = (D * dy - sign_dy * dx * np.sqrt(discriminant)) / dr**2
            sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
            sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

            
            # add currentX and currentY back to the solutions, offset the system back to its original position
            sol1 = [sol_x1 + currentX, sol_y1 + currentY]
            sol2 = [sol_x2 + currentX, sol_y2 + currentY]

            return [sol1, sol2]
        else:
            return None

    def find_target_point(self, intersections, point_ahead, point_behind):

        x1 , y1= point_behind  
        x2 , y2= point_ahead

        intersection_1, intersection_2=intersections
        
        point_validity_1=False
        point_validity_2=False


        # find min and max x y values
        minX = min(x1, x2)
        maxX = max(x1, x2)
        minY = min(y1, y2)
        maxY = max(y1, y2)
            
        # check to see if any of the two solution points are within the correct range
        if (minX <= intersection_1[0] <= maxX and minY <= intersection_1[1] <= maxY):
            point_validity_1=True

        if (minX <= intersection_2[0] <= maxX and minY <= intersection_2[1] <= maxY):
            point_validity_2=True

        if not point_validity_1 and not point_validity_2:
            # no valid intersections → just aim at the original 'ahead' point
            return point_ahead

        elif point_validity_1 and not point_validity_2:
            return intersection_1

        elif not point_validity_1 and point_validity_2:
            return intersection_2

        else:
            
            pt_ahead = np.array(point_ahead)
            pt_int_1   = np.array(intersection_1)            
            pt_int_2   = np.array(intersection_2)

            d1 = np.linalg.norm(pt_ahead - pt_int_1)
            d2 = np.linalg.norm(pt_ahead - pt_int_2)

            return intersection_1 if d1 < d2 else intersection_2
