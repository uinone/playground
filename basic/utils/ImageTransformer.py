import numpy as np
import math
from utils.Interpolator import Interpolator

class ImageTransformer:
    
    def __init__(self, image, interpolator):
        self.image = image.copy()
        self.interpolator = interpolator
        
    def scale(self, scale_width, scale_height):
        frame = ImageTransformer.Padder(self.image).bottom(1).end(1).build()
        
        frame_height, frame_width, frame_channel = self.image.shape
        
        new_frame_grid_height, new_frame_grid_width, new_frame_grid_channel = np.mgrid[:int(frame_height * scale_height), :int(frame_width * scale_width), :3]
        
        new_frame = np.zeros([
            int(frame_height * scale_height), 
            int(frame_width * scale_width), 
            frame_channel
        ], np.int32)
        
        new_frame_grid_height = np.reshape(new_frame_grid_height, -1)
        new_frame_grid_width = np.reshape(new_frame_grid_width, -1)
        new_frame_grid_channel = np.reshape(new_frame_grid_channel, -1)

        scaled_frame_grid_height = new_frame_grid_height * (1/scale_height)
        scaled_frame_grid_width = new_frame_grid_width * (1/scale_width)

        scaled_frame_idx_y = scaled_frame_grid_height.astype(np.int32)
        scaled_frame_idx_x = scaled_frame_grid_width.astype(np.int32)
        
        scaled_frame_shift_y = scaled_frame_grid_height - scaled_frame_idx_y
        scaled_frame_shift_x = scaled_frame_grid_width - scaled_frame_idx_x
        
        points = (frame[scaled_frame_idx_y, scaled_frame_idx_x, new_frame_grid_channel], 
                  frame[scaled_frame_idx_y+1, scaled_frame_idx_x, new_frame_grid_channel], 
                  frame[scaled_frame_idx_y, scaled_frame_idx_x+1, new_frame_grid_channel], 
                  frame[scaled_frame_idx_y+1, scaled_frame_idx_x+1, new_frame_grid_channel])

        new_frame[new_frame_grid_height, new_frame_grid_width, new_frame_grid_channel] = self.interpolator(
            points, 
            scaled_frame_shift_y, 
            scaled_frame_shift_x
        )

        self.image = new_frame
        return self
    
    def rotate(self, rotate_degree, mode="fit"):
        '''
        [mode]
        fit : 회전된 이미지 전부가 출력됩니다.
        naive : 기존 이미지 사이즈에 맞게 회전된 이미지가 출력됩니다.
        그 외 : 입력으로 주어진 이미지가 그대로 반환됩니다.
        '''
        h, w, c = self.image.shape

        frame = ImageTransformer.Padder(self.image).bottom(1).end(1).build()

        rotate_radian = math.radians(rotate_degree)

        rotation_matrix_creator = lambda rad : np.array([
            [math.cos(rad), -math.sin(rad)],
            [math.sin(rad), math.cos(rad)]
        ])

        new_frame_h, new_frame_w = None, None

        rotation_matrix = rotation_matrix_creator(rotate_radian)

        # new_frame 사이즈 측정
        if mode == "fit":
            tmp_rotate_matrix = np.expand_dims(rotation_matrix, 0)

            left_top_point = np.expand_dims(np.array([-w/2, -h/2]), 0)
            left_bottom_point = np.expand_dims(np.array([-w/2, h/2]), 0)
            right_top_point = np.expand_dims(np.array([w/2, -h/2]), 0)
            right_bottom_point = np.expand_dims(np.array([w/2, h/2]), 0)

            points = np.concatenate([
                left_top_point, 
                left_bottom_point, 
                right_top_point, 
                right_bottom_point
            ], axis=0)

            points = np.expand_dims(points, -1)

            rotated_points = (tmp_rotate_matrix @ points).squeeze()

            new_frame_w = int(np.max(rotated_points[:,0]) - np.min(rotated_points[:,0]))
            new_frame_h = int(np.max(rotated_points[:,1]) - np.min(rotated_points[:,1]))

        elif mode == "naive":
            new_frame_h, new_frame_w = h, w

        else:
            return img

        new_grid_h, new_grid_w, new_grid_c = np.mgrid[:new_frame_h, :new_frame_w, :c]

        new_grid_h = np.reshape(new_grid_h, (-1, 1))
        new_grid_w = np.reshape(new_grid_w, (-1, 1))
        new_grid_c = np.reshape(new_grid_c, (-1, 1))

        new_grid = np.concatenate([new_grid_w, new_grid_h], axis=-1)
        new_grid = np.expand_dims(new_grid, -1)

        new_grid[:,0] = new_grid[:,0] - new_frame_w/2
        new_grid[:,1] = new_grid[:,1] - new_frame_h/2

        rotated_grid = (rotation_matrix @ new_grid).squeeze()

        rotated_grid_w, rotated_grid_h = (rotated_grid[:,0] + w/2), (rotated_grid[:,1] + h/2)

        rotated_grid_h_idx, rotated_grid_w_idx = rotated_grid_h.astype(np.int32), rotated_grid_w.astype(np.int32)

        responsible_value = lambda x, max_val: ((x >= 0) & (x < max_val))

        resp_h = responsible_value(rotated_grid_h_idx, h)
        resp_w = responsible_value(rotated_grid_w_idx, w)

        resp = resp_h & resp_w

        new_grid_h = new_grid_h[resp][:,0]
        new_grid_w = new_grid_w[resp][:,0]
        new_grid_c = new_grid_c[resp][:,0]

        rotated_grid_h = rotated_grid_h[resp]
        rotated_grid_w = rotated_grid_w[resp]

        rotated_grid_h_idx = rotated_grid_h_idx[resp]
        rotated_grid_w_idx = rotated_grid_w_idx[resp]

        shift_h = rotated_grid_h - rotated_grid_h_idx
        shift_w = rotated_grid_w - rotated_grid_w_idx

        points = (frame[rotated_grid_h_idx, rotated_grid_w_idx, new_grid_c], 
                  frame[rotated_grid_h_idx+1, rotated_grid_w_idx, new_grid_c], 
                  frame[rotated_grid_h_idx, rotated_grid_w_idx+1, new_grid_c], 
                  frame[rotated_grid_h_idx+1, rotated_grid_w_idx+1, new_grid_c])

        new_frame = np.zeros((new_frame_h, new_frame_w, c), np.uint8)

        new_frame[new_grid_h, new_grid_w, new_grid_c] = self.interpolator(points, shift_h, shift_w)

        self.image = new_frame
        return self
        
    def build(self):
        return self.image
    
    
    # ==================================== [PADDER] ====================================
    class Padder:
        '''
        이미지를 패딩해줍니다.
        빌더 패턴이며, 패딩 요소를 추가한 후 build를 호출해주세요
        '''
        def __init__(self, image):
            self.image = image.copy()

        def _horizontal_padding(self, size, direction="start"):
            '''
            [direction]
            "start" : start padding
            "end" : end padding
            else : return plain image
            '''
            temp = self.image.copy()

            h,w,c = temp.shape

            pad = np.zeros((h, size, c), temp.dtype)

            if direction == "start":
                temp = np.concatenate([pad, temp], axis=1)
            elif direction == "end":
                temp = np.concatenate([temp, pad], axis=1)

            return temp

        def _vertical_padding(self, size, direction="top"):
            '''
            [direction]
            "top" : top padding
            "bottom" : bottom padding
            else : return plain image
            '''
            temp = self.image.copy()

            h,w,c = temp.shape

            pad = np.zeros((size, w, c), temp.dtype)

            if direction == "top":
                temp = np.concatenate([pad, temp], axis=0)
            elif direction == "bottom":
                temp = np.concatenate([temp, pad], axis=0)

            return temp

        def top(self, size):
            self.image = self._vertical_padding(size, "top")
            return self

        def bottom(self, size):
            self.image = self._vertical_padding(size, "bottom")
            return self

        def start(self, size):
            self.image = self._horizontal_padding(size, "start")
            return self

        def end(self, size):
            self.image = self._horizontal_padding(size, "end")
            return self

        def build(self):
            return self.image