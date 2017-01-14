
import numpy as np
from scipy.ndimage.morphology import binary_erosion

start_field = (20, 34)
end_field = (140, 194)
field_size = (160, 160)
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
    # preprocessing
    field = img[start_field[1]:end_field[1], :, :]
    background_color = np.array([np.median(field[:, :, 0]), np.median(field[:, :, 1]), np.median(field[:, :, 2])])
    binary_field = np.any(field != background_color, 2)
    columns = np.where(np.any(binary_field, 0))[0]

    # find right bar
    left_bar = np.empty(2)
    left_bar[0] = np.mean(bar_left_columns)
    left_bar[1] = 79.5
    right_bar = np.empty(2)
    right_bar[0] = np.mean(bar_left_columns)
    right_bar[1] = 79.5
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
    ball_dicrection = np.array([0, 0])
    if np.sum(erosion):
        assert np.sum(erosion) in [1, 2] # no other object detected
        ball[0] = np.mean(np.where(np.any(erosion, 0))[0])
        ball[1] = np.mean(np.where(np.any(erosion, 1))[0])
        if np.any(np.isnan(last_ball_position)) == False:
            ball_dicrection[0] = ball[0] - last_ball_position[0]
            ball_dicrection[1] = ball[1] - last_ball_position[1]
        last_ball_position[0] = ball[0]
        last_ball_position[1] = ball[1]
    else: # place ball in front of oponnent (left bar)
        ball[0] = left_bar[0] + 2
        ball[1] = left_bar[1]
        last_ball_position = np.array([np.NaN, np.NaN])


    distance_between_ball_and_left_bar = np.sqrt((ball[0]-right_bar[0])**2 + (ball[1]-right_bar[1])**2)

    return {'left_bar': left_bar, 'right_bar': right_bar, \
    'ball': ball, 'distance': distance_between_ball_and_left_bar, \
    'ball_dicrection': ball_dicrection, 'last_ball_position': last_ball_position}



def get_state(position_dict, add_direction=False):
    '''
    Given the psotions dict, it returns a subset of normalized positions.
    In addition it can add the normalized direction vector
    '''
    # center field and normalize to -1..1
    position_dict['right_bar'] = (position_dict['right_bar'] - 80) / 80.0
    position_dict['left_bar'] = (position_dict['left_bar'] - 80) / 80.0
    position_dict['ball'] = (position_dict['ball'] - 80) / 80.0

    state = np.array([position_dict['right_bar'][1], \
        position_dict['left_bar'][0], \
        position_dict['ball'][0], \
        position_dict['ball'][1]])

    if add_direction:
        position_dict['ball_dicrection'] = position_dict['ball_dicrection'] / 80.0
        state = np.concatenate(state, position_dict['ball_dicrection'])


    # y position of self, opponent and ball (x,y) position
    return state