import numpy as np
from scipy.ndimage.morphology import binary_erosion
import matplotlib.pyplot as plt


start_field = (20, 34)
end_field = (140, 194)
bar_right_columns = np.array([140, 141, 142, 143])
bar_left_columns = np.array([16, 17, 18, 19])

def get_positions(img, last_ball_position=np.array([np.NaN, np.NaN])):
    '''
    Parses the akari RGB images and returns a dict with:
     - left/right bar position
     - ball position
     - distance between balla nd right bar (our agent)
    If last_ball_position is given, it will also return ball_dicrection
    '''

    # default positions
    left_bar = np.empty(2)
    left_bar[0] = np.mean(bar_left_columns)
    left_bar[1] = 88.5
    right_bar = np.empty(2)
    right_bar[0] = np.mean(bar_right_columns)
    right_bar[1] = 69.5
    ball = np.array([76, 84])
    ball_dicrection = np.array([0, 0])

    # preprocessing
    field = img[start_field[1]:end_field[1], :, :]
    background_color = np.array([np.median(field[:, :, 0]), np.median(field[:, :, 1]), np.median(field[:, :, 2])])
    binary_field = np.any(field != background_color, 2)
    columns = np.where(np.any(binary_field, 0))[0]

    if np.any(background_color != [109, 118, 43]): # first frame
        # erosion elements
        bar_element = np.empty((1, 4))
        bar_element[:] = True

        # left bar
        verify_left_bar_is_there = np.vectorize(lambda x: np.any(x == bar_left_columns))
        if np.sum(verify_left_bar_is_there(columns)) == 4:
            left_bar[0] = np.mean(bar_left_columns)
            # bar is only object with 4x1 (full size 4x16)
            erosion = binary_erosion(binary_field[:, bar_left_columns], bar_element)
            positions = np.where(erosion[:, 2])[0]

            if positions.size == 16:
                left_bar[1] = np.mean(positions)
            else:
                if positions[0] < binary_field.shape[0]/2: # upper part is clipped
                    left_bar[1] = positions[-1] - 7.5
                else: # lower part is clipped
                    left_bar[1] = positions[0] + 7.5

        # right bar
        verify_right_bar_is_there = np.vectorize(lambda x: np.any(x == bar_right_columns))
        if np.sum(verify_right_bar_is_there(columns)) == 4:
            right_bar[0] = np.mean(bar_right_columns)
            # bar is only object with 4x1 (full size 4x16)
            erosion = binary_erosion(binary_field[:, bar_right_columns], bar_element)
            positions = np.where(erosion[:, 2])[0]

            if positions.size == 16:
                right_bar[1] = np.mean(positions)
            else:
                if positions[0] < binary_field.shape[0]/2: # upper part is clipped
                    right_bar[1] = positions[-1] - 7.5
                else: # lower part is clipped
                    right_bar[1] = positions[0] + 7.5

        # ball
        ball_field = np.all(field == [236, 236, 236], 2)
        assert np.sum(ball_field) <= 10
        if np.sum(ball_field) > 0:
            ball[0] = np.mean(np.where(np.any(ball_field, 0))[0])
            ball[1] = np.mean(np.where(np.any(ball_field, 1))[0])
            if not np.any(np.isnan(last_ball_position)):
                if np.sum(np.abs(ball - last_ball_position)) < 50:
                    ball_dicrection[0] = ball[0] - last_ball_position[0]
                    ball_dicrection[1] = ball[1] - last_ball_position[1]

        elif not np.any(np.isnan(last_ball_position)): # ball might be hidden while colliding with end of field
            ball[0] = last_ball_position[0]
            ball[1] = last_ball_position[1]

    distance_between_ball_and_left_bar = right_bar[0] - ball[0]

    if np.abs(ball[1]-right_bar[1]) >= 20 or ball_dicrection[0] >= 0:
        distance_between_ball_and_left_bar = np.Inf

    if False and distance_between_ball_and_left_bar < 8:
        # overlay information
        print distance_between_ball_and_left_bar, ball_dicrection
        img[start_field[1]+left_bar[1], left_bar[0], :] = 0.5
        img[start_field[1]+right_bar[1], right_bar[0], :] = 0.5
        img[start_field[1]+ball[1], ball[0], :] = 0.5
        plt.imshow(img)
        plt.show()

    return {'left_bar': left_bar, 'right_bar': right_bar, \
        'ball': ball, 'distance': distance_between_ball_and_left_bar, \
        'ball_dicrection': ball_dicrection}



def get_state(position_dict, add_direction=False):
    '''
    Given the psotions dict, it returns a subset of normalized positions.
    In addition it can add the normalized direction vector
    '''
    # center field and normalize to -1..1
    # y position of self, opponent and ball (x,y) position
    state = np.array([(position_dict['right_bar'][1]-80)/80.0, \
        (position_dict['left_bar'][1]-80)/80.0, \
        (position_dict['ball'][0]-80)/80.0, \
        (position_dict['ball'][1]-80)/80.0])

    if add_direction:
        state = np.concatenate((state, position_dict['ball_dicrection']/80.0))

    return state