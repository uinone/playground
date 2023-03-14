import numpy as np

class Interpolator:
    def bilinear(points, x, y):
        '''
        points : [[left_top_points], 
                  [left_bottom_points], 
                  [right_top_points], 
                  [right_bottom_points]]
        x : amount of shifting in horizontal axis
        y : amount of shifting in vertical axis
        '''
        left_top_points, left_bottom_points, right_top_points, right_bottom_points = points[0], points[1], points[2], points[3]

        left_side_points = (1-x) * left_top_points + x * left_bottom_points
        right_side_points = (1-x) * right_top_points + x * right_bottom_points

        return (1-y) * left_side_points + y * right_side_points