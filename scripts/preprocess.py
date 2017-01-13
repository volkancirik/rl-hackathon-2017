
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.morphology import binary_erosion

start_field = (20, 34)
end_field = (140, 194)
field_size = (160, 160)
bar_right_columns = np.array([140, 141, 142, 143])
bar_left_columns = np.array([16, 17, 18, 19])

def get_positions(img):
    # preprocessing
    field = img[start_field[1]:end_field[1], :, :]
    background_color = np.array([np.median(field[:, :, 0]), np.median(field[:, :, 1]), np.median(field[:, :, 2])])
    binary_field = np.any(field != background_color, 2)
    rows = np.where(np.any(binary_field, 1))[0]
    columns = np.where(np.any(binary_field, 0))[0]

    # find right bar
    left_bar = np.empty(2)
    left_bar[:] = np.nan
    right_bar = np.empty(2)
    right_bar[:] = np.nan
    ball = np.empty(2)
    ball[:] = np.nan

    # erosion elements
    bar_element = np.empty((1, 4))
    bar_element[:] = True
    ball_element = np.empty((4, 2)) # is never clipped
    ball_element[:] = True

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
            if positions[0] < binary_field.shape[0]/2: # lower part is clipped
                left_bar[1] = positions[0] + 7.5
            else: # upper part is clipped
                left_bar[1] = positions[-1] + 7.5

        # remove from binary_field -> easier to find ball
        binary_field[np.ix_(positions, bar_left_columns)] = False

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
            if positions[0] < binary_field.shape[0]/2: # lower part is clipped
                right_bar[1] = positions[0] + 7.5
            else: # upper part is clipped
                right_bar[1] = positions[-1] + 7.5

        # remove from binary_field -> easier to find ball
        binary_field[np.ix_(positions, bar_right_columns)] = False

    # ball (only remaining object (if present))
    erosion = binary_erosion(binary_field, ball_element)
    if np.sum(erosion):
        assert np.sum(erosion) == 1 # no other object detected
        ball[0] = np.where(np.any(erosion, 0))[0][0]
        ball[1] = np.where(np.any(erosion, 1))[0][0]

    return {'left_bar': left_bar, 'right_bar': right_bar, 'ball': ball}



def get_state(position_dict):
    # center field and normalize to -1..1
    position_dict['right_bar'] = (position_dict['right_bar'] - 80) / 80.0
    position_dict['ball'] = (position_dict['ball'] - 80) / 80.0

    # y position of self (right bar) and ball (x,y) position
    return (position_dict['right_bar'][1], \
        position_dict['ball'][0], \
        position_dict['ball'][1])