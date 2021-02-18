import numpy as np

class Grids():
    def __init__(self, grid_size, height, width):
        self.size = grid_size
        self.number = None
        self.center_lst = None
        self.lst = None
        self.topLeft_lst = None
        self.botRight_lst = None
        self.constructGridInfo(grid_size, height, width)

    def constructGridInfo(self, grid_size, height, width):
        '''
        Cut the target image into many grids.
        construct 1) list consists of each grid's pixel's coordinates.
                  2) list consists of each grid's center's coordinate.
        The lists both consist of N elements, where N is number of grids.

        '''
        grid_lst = []  # a list consists of grid's pixel's coordinate.
        grid_center_lst = []  # a list consists of each grid's center coordinate.
        grid_topleft_lst = []
        grid_botright_lst = []
        x_grid_interval = np.arange(0, width, grid_size)
        y_grid_interval = np.arange(0, height, grid_size)

        # calculate every grid's top left coordinate.
        x_grids_top_left, y_grids_top_left = np.meshgrid(
            x_grid_interval, y_grid_interval)

        # calculate pixel's coordinate in each grid.
        # The loop will run N times, N is total number of grids.
        for start_x, start_y in zip(x_grids_top_left.flatten(), y_grids_top_left.flatten()):

            '''
            Calculate each grid's bottom right coordinate.
            Note: 
            If end_x=5 end_y=5, then the grid's bottom right coordinate is (4,4)!!!!!!!!!
            '''
            end_x = start_x + grid_size if ((start_x+grid_size) < width) else width
            end_y = start_y + grid_size if ((start_y+grid_size) < height) else height

            grid_center = np.array(((end_x+start_x)//2, (end_y+start_y)//2))
            grid_center_lst.append(grid_center)
            grid_topleft = np.array( (start_x, start_y))
            grid_topleft_lst.append(grid_topleft)
            grid_botright = np.array( (end_x-1, end_y-1))
            grid_botright_lst.append(grid_botright)


            x_coordi, y_coordi = np.meshgrid(
                np.arange(start_x, end_x), np.arange(start_y, end_y))

            x_coordi = x_coordi.flatten()
            y_coordi = y_coordi.flatten()

            x_y_coordi = np.stack((x_coordi, y_coordi))
            grid_lst.append(x_y_coordi)

        self.lst = grid_lst
        self.center_lst = grid_center_lst
        self.botRight_lst = grid_botright_lst
        self.topLeft_lst = grid_topleft_lst
        self.number = len(grid_lst)