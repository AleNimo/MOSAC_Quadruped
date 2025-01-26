import numpy as np
terrain_x_size = 20          #meters
terrain_y_size = 20          #meters
terrain_resolution = 0.05   #meters

terrain_x_dimension = int(terrain_x_size/terrain_resolution)
terrain_y_dimension = int(terrain_y_size/terrain_resolution)

out_string = ''

for x_index in range(terrain_x_dimension):
    for y_index in range(terrain_y_dimension):

        out_string = out_string + "%.3f" % np.random.uniform(0, 0.01) + ","

        # if x_index % 3 == 0 or y_index % 3 == 0:
        #     out_string = out_string + "-0.01,"
        # else:
        #     out_string = out_string + "0,"

print(out_string)