import numpy as np
import cv2
from configs.clrnet.clr_resnet34_tusimple import world_x_interval, world_y_interval

# birdeyeview를 만드는 객체
class IBE(object):
    def __init__(self, extrinsic, intrinsic):

        self.world_x_max = 11
        self.world_x_min = 0.5 # height
        self.world_y_max = 4
        self.world_y_min = -4 # width

        self.world_x_interval = world_x_interval # height
        self.world_y_interval = world_y_interval # width 작을수록 map이 커짐

        # world_x와 y의 interval을 바꾸면 get_curve의 get_steer와 
        # points_to_gps의 points2no_z_point도 바꿔줘야한다.
        
        # clrnet configs에서 변경가능


        """
        BEV 이미지에서 행 방향으로 1 픽셀 만큼 증감하면 world 좌표계에서 X 방향으로 0.05 (m) 만큼 증감하도록 하고 
        BEV 이미지에서 열 방향 1 픽셀 만큼 증가하면 world 좌표계에서 Y 방향으로 0.025 (m) 만큼 증감하도록 만들 수 있습니다.
        """
        
        print("world_x_min : ", self.world_x_min)
        print("world_x_max : ", self.world_x_max)
        print("world_x_interval (m) : ", self.world_x_interval)
        print()
        
        print("world_y_min : ", self.world_y_min)
        print("world_y_max : ", self.world_y_max)
        print("world_y_interval (m) : ", self.world_y_interval)

        # Calculate the number of rows and columns in the output image
        self.output_width = int(np.ceil((self.world_y_max - self.world_y_min) / self.world_y_interval))
        self.output_height = int(np.ceil((self.world_x_max - self.world_x_min) / self.world_x_interval))

        print("(width, height) :", "(", self.output_width, ",",  self.output_height, ")")

        print("create map...")
        self.map_x, self.map_y = self.generate_direct_backward_mapping(self.world_x_min, self.world_x_max, self.world_x_interval, self.world_y_min, self.world_y_max, self.world_y_interval, extrinsic, intrinsic)
        print("complete map!")
        

    def generate_direct_backward_mapping(self,
    world_x_min, world_x_max, world_x_interval, 
    world_y_min, world_y_max, world_y_interval, extrinsic, intrinsic):
    
        print("world_x_min : ", world_x_min)
        print("world_x_max : ", world_x_max)
        print("world_x_interval (m) : ", world_x_interval)
        print()
        
        print("world_y_min : ", world_y_min)
        print("world_y_max : ", world_y_max)
        print("world_y_interval (m) : ", world_y_interval)
        
        world_x_coords = np.arange(world_x_max, world_x_min, -world_x_interval)
        world_y_coords = np.arange(world_y_max, world_y_min, -world_y_interval)
        
        output_height = len(world_x_coords)
        output_width = len(world_y_coords)
        
        map_x = np.zeros((output_height, output_width)).astype(np.float32)
        map_y = np.zeros((output_height, output_width)).astype(np.float32)
        
        for i, world_x in enumerate(world_x_coords):
            for j, world_y in enumerate(world_y_coords):
                # world_coord : [world_x, world_y, 0, 1]
                # uv_coord : [u, v, 1]
                
                world_coord = [world_x, world_y, 0, 1]
                
                camera_coord = extrinsic[:3, :] @ world_coord
                uv_coord = intrinsic[:3, :3] @ camera_coord
                uv_coord /= uv_coord[2]

                # map_x : (H, W)
                # map_y : (H, W)
                # dst[i][j] = src[ map_y[i][j] ][ map_x[i][j] ]

                map_x[i][j] = int(np.round(uv_coord[0]))
                map_y[i][j] = int(np.round(uv_coord[1]))

                # i -> 0 ~ 480 height
                # j -> 0 ~ 640 width

        return map_x, map_y
    

    def birdeyeview(self, image):

        output_image = cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) 
        
        return output_image