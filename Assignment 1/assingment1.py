import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import functions

###################### Part 1 ###############################
## Calculating points and vectors

# Iliac Crest (Pelvis)
il_crest = [-1063, 955.7]

# Greater Trochanter (Hip)
g_troch = [-1113, 823.9]

# Lateral plateau (knee)
lat_plateau = [-1254, 433.8]

# Lateral Malleolus (Ankle)
lat_malleolus = [-1523.84, 178.9]

# Calling fuction to calculate vectors
trunk_vector = functions.single_calc_pos_vector(il_crest, g_troch)
thigh_vector = functions.single_calc_pos_vector(g_troch, lat_plateau)
shank_vector = functions.single_calc_pos_vector(lat_plateau, lat_malleolus)
print("Position Vectors:")
print(f"Trunk Vector: {trunk_vector}\nThigh Vector: {thigh_vector}\nShank Vector: {shank_vector}")

## Caluclating angles
# Calling function to calculate angles
trunk_angle = functions.single_calc_angle(g_troch, il_crest)
thigh_angle = functions.single_calc_angle(lat_plateau, g_troch)
shank_angle = functions.single_calc_angle(lat_malleolus, lat_plateau)
print("Segment Angles:")
print(f"Trunk Angle: {trunk_angle:.2f} degrees\nThigh Angle: {thigh_angle:.2f} degrees\nShank Angle: {shank_angle:.2f} degrees")

# Calculating joint angles
hip_angle = trunk_angle - thigh_angle
knee_angle = thigh_angle - shank_angle

print("Joint Angles:")
print(f"Hip Angle: {hip_angle:.2f} degrees\nKnee Angle: {knee_angle:.2f} degrees")

# Plotting points and vectors to check work
plt.scatter(il_crest[0], il_crest[1], color='blue')
plt.scatter(g_troch[0], g_troch[1], color='orange')
plt.scatter(lat_plateau[0], lat_plateau[1], color='green')
plt.scatter(lat_malleolus[0], lat_malleolus[1], color='red')
plt.quiver(il_crest[0], il_crest[1], trunk_vector[0], trunk_vector[1], angles='xy', scale_units='xy', scale=1, color='blue')
plt.quiver(g_troch[0], g_troch[1], thigh_vector[0], thigh_vector[1], angles='xy', scale_units='xy', scale=1, color='orange')
plt.quiver(lat_plateau[0], lat_plateau[1], shank_vector[0], shank_vector[1], angles='xy', scale_units='xy', scale=1, color='green')
plt.xlim(-1600, -900)
plt.legend(['Iliac Crest', 'Greater Trochanter', 'Lateral Plateau', 'Lateral Malleolus'])

###################### Part 2 ###############################
# Reading in data
data_df = functions.format_df("KIN530_2D_Kinematics_2025.csv", "KIN530_2D_DataHeaders_2025.csv")
alt_df = functions.format_df("KIN530_Data_Alternate.csv", "KIN530_2D_DataHeaders_2025.csv")

joint_list = ['hip', 'knee', 'ankle']

# Calculating kinematics
data_df = functions.calculate_kinematics(data_df, joint_list)
alt_data_df = functions.calculate_kinematics(alt_df, joint_list)

# Plotting
# Defining time vector
original_time = data_df['Time']
alt_time = alt_data_df['Time']

# Locating heel strikes
heel_strikes = functions.locate_heel_strikes(data_df)
alt_heel_strikes = functions.locate_heel_strikes(alt_data_df)

# Plotting heel vertical velo
plt.plot(original_time, data_df['heel_velo'])
plt.plot(alt_time, alt_data_df['heel_velo'], color='orange', alpha=0.5)
# plt.plot(time, data_df['heel_Z'])
plt.plot([original_time[i] for i in heel_strikes], 
         [data_df['heel_velo'][i] for i in heel_strikes],
         "ro", label='Heel Strikes')
plt.plot([alt_time[i] for i in alt_heel_strikes],
            [alt_data_df['heel_velo'][i] for i in alt_heel_strikes],
            "mo", label='Alt Heel Strikes', alpha=0.5)
plt.title('Heel Vertical Velocity')
plt.ylabel('Velocity (m/s)')
plt.xlabel('Time (s)')
plt.legend()
plt.title('Heel Vertical Velocity')
print(f" Heel Strikes Occur at Times: {[float(data_df.loc[i, 'Time']) for i in heel_strikes]} seconds")

# Plotting joint angles
fig, (ax1, ax2) = plt.subplots(2, 1)
# Angles subplot
# Original data
ax1.plot(original_time, data_df['hip_angle'], label='Original Hip')
ax1.plot(original_time, data_df['knee_angle'], color='r', label='Original Knee')
ax1.axvline(original_time[heel_strikes[0]], color='b', linestyle='--')
ax1.text(original_time[heel_strikes[0]] + 0.01,  .98*max(data_df['knee_angle']), 
         'Org Heel Strike', color='b', verticalalignment='top')
ax1.text(-0.05, 0.92*max(alt_data_df['knee_angle']), "Flexion", horizontalalignment='left', verticalalignment='bottom')
ax1.text(-0.05, 0.83*min(alt_data_df['hip_angle']), "Extension", horizontalalignment='left', verticalalignment='top')


# Alt data
ax1.plot(alt_time, alt_data_df['hip_angle'], color='c', linestyle = '--', label='Alt Hip', alpha=0.3)
ax1.plot(alt_time, alt_data_df['knee_angle'], color='m', linestyle = '--', label='Alt Knee', alpha=0.3)
ax1.axvline(alt_time[alt_heel_strikes[0]], color='black', linestyle='--', alpha=0.5)
ax1.text(alt_time[alt_heel_strikes[0]] + 0.01,  .88*max(data_df['knee_angle']), 
         'Alt Heel Strike', color='black', verticalalignment='top', alpha=0.5)
ax1.set_ylabel('Angle (degrees)')
ax1.set_xlabel('Time (s)')
ax1.legend(loc='upper right')
ax1.set_title('Angles')
ax1.grid()

# Velocities subplot
# Original data
ax2.plot(original_time, data_df['knee_velocity'], label='Original Knee')
ax2.axvline(original_time[heel_strikes[0]], color='b', linestyle='--')
ax2.text(original_time[heel_strikes[0]] + 0.01, max(data_df['knee_velocity']), 
         'Org Heel Strike', color='b', verticalalignment='top')
ax2.text(-0.05, 0.92*max(data_df['knee_velocity']), "Flexion", horizontalalignment='left', verticalalignment='bottom')
ax2.text(-0.05, 0.92*min(data_df['knee_velocity']), "Extension", horizontalalignment='left', verticalalignment='top')

# Alt data
ax2.plot(alt_time, alt_data_df['knee_velocity'], color='m', linestyle = '--', label='Alt Knee', alpha=0.3)
ax2.axvline(alt_time[alt_heel_strikes[0]], color='black', linestyle='--', alpha=0.5)
ax2.text(alt_time[alt_heel_strikes[0]] + 0.01, .86*max(data_df['knee_velocity']), 
         'Alt Heel Strike', color='black', verticalalignment='top', alpha=0.5)
ax2.set_title('Knee Angular Velocity')
ax2.set_ylabel('Velocity (deg/s)')
ax2.set_xlabel('Time (s)')
ax2.legend(loc='upper right')
ax2.grid()
plt.tight_layout()
plt.suptitle('Joint Angles and Velocities', y=1.02)
plt.savefig('Joint_Angles_and_Velocities.png', bbox_inches='tight')
plt.show()