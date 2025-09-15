import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Defining function to calc segment position vectors
def single_calc_pos_vector(point1, point2):
    # x2-x1, y2-y1
    return [point2[0] - point1[0], point2[1] - point1[1]]

# Defining function to calc segment angles
def single_calc_angle(distal_point, proximal_point):
    #Using equation from class
    angle = np.arctan2(proximal_point[1] - distal_point[1], proximal_point[0] - distal_point[0])
    # Converting to degrees
    angle_deg = np.degrees(angle)
    return angle_deg

# Creating function to create to load in data
def format_df(data_csv, header_csv):
    # Reading in original data
    header_csv = pd.read_csv(header_csv, header=None)
    df = pd.read_csv(data_csv, header=None)

    #Creating headers from header file
    df.columns = header_csv.iloc[0]
    # Formatting headers to remove spaces and adding column titles
    df.columns = df.columns.str.replace(' ', '_')
    df.to_csv("KIN530_2D_Kinematics_2025_Formatted.csv", index=False)

    # Converting millimeters to meters
    excluded_col = 'Time'
    df_to_convert = df.drop(columns=[excluded_col])
    df_to_convert = df_to_convert * .001

    df = pd.concat([df[excluded_col], df_to_convert], axis=1)
    return df

# Changing funcntion to accept df inputs
def calc_angle(distal_point, proximal_point):
    #Using equation from class
    angle = np.arctan2(proximal_point.iloc[1] - distal_point.iloc[1], proximal_point.iloc[0] - distal_point.iloc[0])
    # Converting to degrees
    angle_deg = np.degrees(angle)
    return angle_deg

def calc_pos_vector(point1, point2):
    # x2-x1, y2-y1
    return [point2.iloc[0] - point1.iloc[0], point2.iloc[1] - point1.iloc[1]]

def pairing_points(df, point_name):
    coordinates = df[[f'{point_name}_X', f'{point_name}_Z']]
    return coordinates

# Creating function to calculate kinematics
def calculate_kinematics(df, joint_list):
    # Pairing points for each segment
    il_crest_vec = pairing_points(df, 'il_crest')
    grt_troc_vec = pairing_points(df,'grt_troc')
    lat_con_vec = pairing_points(df,'lat_con')
    lat_mal_vec = pairing_points(df,'lat_mall')
    heel_vec = pairing_points(df, 'heel')
    fifth_mtar_vec = pairing_points(df, '5th_mtar')

    for time_sample in df.index:
        # Calculate segment angles
        df.loc[time_sample, 'trunk_angle'] = calc_angle(grt_troc_vec.iloc[time_sample], il_crest_vec.iloc[time_sample])
        df.loc[time_sample, 'thigh_angle'] = calc_angle(lat_con_vec.iloc[time_sample], grt_troc_vec.iloc[time_sample])
        df.loc[time_sample,'shank_angle'] = calc_angle(lat_mal_vec.iloc[time_sample], lat_con_vec.iloc[time_sample])
        df.loc[time_sample,'foot_angle'] = calc_angle(heel_vec.iloc[time_sample], lat_mal_vec.iloc[time_sample])

        # Calculate joint angles
        df.loc[time_sample,'hip_angle'] = df.loc[time_sample,'thigh_angle'] - df.loc[time_sample,'trunk_angle']
        df.loc[time_sample,'knee_angle'] = df.loc[time_sample,'thigh_angle'] - df.loc[time_sample,'shank_angle']
        df.loc[time_sample,'ankle_angle'] = df.loc[time_sample,'shank_angle'] - df.loc[time_sample,'foot_angle']

        # Calculating joint velocities
        for joint in joint_list:
            if time_sample == 0:
                df.loc[time_sample, f"{joint}_velocity"] = ((df.loc[(time_sample + 1), f"{joint}_angle"]) - df.loc[time_sample, f"{joint}_angle"]) / ((df.loc[(time_sample + 1), 'Time']) - df.loc[time_sample, 'Time'])
            elif time_sample == df.index[-1]:
                df.loc[time_sample, f"{joint}_velocity"] = ((df.loc[time_sample, f"{joint}_angle"]) - df.loc[(time_sample - 1), f"{joint}_angle"]) / ((df.loc[time_sample, 'Time']) - df.loc[(time_sample - 1), 'Time'])
            else:
                df.loc[time_sample, f"{joint}_velocity"] = ((df.loc[(time_sample + 1), f"{joint}_angle"]) - df.loc[(time_sample - 1), f"{joint}_angle"]) / ((df.loc[(time_sample + 1), 'Time']) - df.loc[(time_sample - 1), 'Time'])
                
        # Calculating heel vertical veloctiy + acceleration to find heel strike
        if time_sample == 0:
            df.loc[time_sample, 'heel_velo'] = ((df.loc[(time_sample + 1), 'heel_Z']) - df.loc[time_sample, 'heel_Z']) / ((df.loc[(time_sample + 1), 'Time']) - df.loc[time_sample, 'Time'])
        elif time_sample == df.index[-1]:
            df.loc[time_sample, 'heel_velo'] = ((df.loc[time_sample, 'heel_Z']) - df.loc[(time_sample - 1), 'heel_Z']) / ((df.loc[time_sample, 'Time']) - df.loc[(time_sample - 1), 'Time'])
        else:
            df.loc[time_sample, 'heel_velo'] = ((df.loc[(time_sample + 1), 'heel_Z']) - df.loc[(time_sample - 1), 'heel_Z']) / ((df.loc[(time_sample + 1), 'Time']) - df.loc[(time_sample - 1), 'Time'])

        # Calculating heel vertical acceleration
        if time_sample == 0:
            df.loc[time_sample, 'heel_accel'] = ((df.loc[(time_sample + 1), 'heel_velo']) - df.loc[time_sample, 'heel_velo']) / ((df.loc[(time_sample + 1), 'Time']) - df.loc[time_sample, 'Time'])
        elif time_sample == df.index[-1]:
            df.loc[time_sample, 'heel_accel'] = ((df.loc[time_sample, 'heel_velo']) - df.loc[(time_sample - 1), 'heel_velo']) / ((df.loc[time_sample, 'Time']) - df.loc[(time_sample - 1), 'Time'])
        else:
            df.loc[time_sample, 'heel_accel'] = ((df.loc[(time_sample + 1), 'heel_velo']) - df.loc[(time_sample - 1), 'heel_velo']) / ((df.loc[(time_sample + 1), 'Time']) - df.loc[(time_sample - 1), 'Time'])
    return df

# Creating function to locate heel strikes
def locate_heel_strikes(df):
    # Finding local minima from heel velo (heel strikes)
    # Must negate heel velo, since scipy "find_peaks" function finds local maxima
    local_minima_idx, properties = find_peaks(df['heel_velo']*-1, height=0)

    # Getting magnitudes of the local minima
    peak_magnitudes = properties['peak_heights']
    peak_info = list(zip(peak_magnitudes, local_minima_idx))

    # Finding the local minima with the steepest decline (should be the heel strikes)
    # Sorting by magnitudes (index 0 of the tuple created by peak_info)
    peak_info.sort(key=lambda x: x[0], reverse=True)

    # creating just a list of sorted minima, but only using those when the heel is close to the ground
    sorted_minima = [index for magnitude, index in peak_info if df.loc[index, 'heel_Z'] < 0.06]
    # defining heel strikes as the highest 2 values
    heel_strikes = sorted_minima[:2]
    # reversinig list to be in chronological order
    heel_strikes.reverse()
    return heel_strikes
