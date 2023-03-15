import numpy as np

class Interpolator:
    
    class Bilinear:
        
        def __call__(self, points, shift_h, shift_w):
            '''
            points : [[left_top_points], 
                      [left_bottom_points], 
                      [right_top_points], 
                      [right_bottom_points]]
            shift_h : amount of shifting in vertical axis
            shift_w : amount of shifting in horizontal axis
            '''
            left_top_points, left_bottom_points, right_top_points, right_bottom_points = points[0], points[1], points[2], points[3]

            left_side_points = (1-shift_w) * left_top_points + shift_w * left_bottom_points
            right_side_points = (1-shift_w) * right_top_points + shift_w * right_bottom_points

            return (1-shift_h) * left_side_points + shift_h * right_side_points
        
    class Triangular:
        
        def _get_abgam(self, A, dot1, dot2, center_dot):
            A = np.expand_dims(A, 0)
            A = np.linalg.pinv(A)

            b = np.concatenate([dot1, dot2, center_dot], -1)
            b = np.expand_dims(b, -1)

            return (A @ b)

        def _get_value(self, abg, shift_h, shift_w):
            sh = np.expand_dims(shift_h, -1)
            sw = np.expand_dims(shift_w, -1)
            one = np.expand_dims(np.ones_like(shift_h), -1)

            l = np.concatenate([sh, sw, one], -1)
            l = np.expand_dims(l, -2)

            return np.reshape((l @ abg), -1)

        def __call__(self, points, shift_w, shift_h):
            center_dot = np.expand_dims(np.mean(points, 0), -1)

            interpolated = np.zeros_like(points[0])

            lt = np.expand_dims(points[0], -1)
            lb = np.expand_dims(points[1], -1)
            rt = np.expand_dims(points[2], -1)
            rb = np.expand_dims(points[3], -1)

            A1 = np.array([[0, 0, 1],
                           [0, 1, 1],
                           [0.5, 0.5, 1]])

            A2 = np.array([[0, 1, 1],
                           [1, 1, 1],
                           [0.5, 0.5, 1]])

            A3 = np.array([[1, 0, 1],
                           [1, 1, 1],
                           [0.5, 0.5, 1]])

            A4 = np.array([[0, 0, 1],
                           [1, 0, 1],
                           [0.5, 0.5, 1]])

            case1 = (shift_w <= shift_h) & (shift_w < (-shift_w+1)) # (0,0) - (0,1) 평면
            case2 = (shift_w < shift_h) & (shift_h >= (-shift_w+1)) # (0,1) - (1,1) 평면
            case3 = (shift_w >= shift_h) & (shift_h > (-shift_w+1)) # (1,0) - (1,1) 평면
            case4 = (shift_w > shift_h) & (shift_h <= (-shift_w+1)) # (0,0) - (1,0) 평면


            interpolated[case1] = self._get_value(
                self._get_abgam(
                    A1, 
                    lt[case1], 
                    lb[case1], 
                    center_dot[case1]
                ), 
                shift_h[case1], 
                shift_w[case1]
            )

            interpolated[case2] = self._get_value(
                self._get_abgam(
                    A2, 
                    lb[case2], 
                    rb[case2], 
                    center_dot[case2]
                ), 
                shift_h[case2], 
                shift_w[case2]
            )

            interpolated[case3] = self._get_value(
                self._get_abgam(
                    A3, 
                    rt[case3], 
                    rb[case3], 
                    center_dot[case3]
                ), 
                shift_h[case3], 
                shift_w[case3]
            )

            interpolated[case4] = self._get_value(
                self._get_abgam(
                    A4, 
                    lt[case4], 
                    rt[case4], 
                    center_dot[case4]
                ), 
                shift_h[case4], 
                shift_w[case4]
            )
            
            case5 = interpolated == 0
            interpolated[case5] = self._get_value(
                self._get_abgam(
                    A1, 
                    lt[case5], 
                    lb[case5], 
                    center_dot[case5]
                ), 
                shift_h[case5], 
                shift_w[case5]
            )

            return interpolated
    