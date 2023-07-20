import numpy as np

# coil description: Cylindrical unshielded coil

plot_all = 1  # set to 1, to optionally plot intermediate steps

# define coil-parameters of the matrix coil: segments_angular, half_length, len_step
CoilDefinition = {}
CoilDefinition['Partitions'] = 2
segments_angular = 48
segments_angular_shield = segments_angular
half_length = 0.6  # 600mm
len_step = 0.015  # 15mm
r_coil = 0.4  # 400mm coil diameter
r_shield = 0.5

arc_angle = 360 / segments_angular
elm_angle, elm_z = np.meshgrid(np.arange(0, segments_angular) * arc_angle, np.arange(-half_length, half_length + len_step, len_step))
CoilDefinition[1] = {'num_elements': elm_angle.shape}
elm_angle_shift = np.roll(elm_angle, -1, axis=0)


import numpy as np

# Define Cylindrical Main Surface

CoilDefinition[1]['thin_wire_nodes_start'] = np.column_stack([np.cos(np.radians(elm_angle.flatten())) * r_coil, np.sin(np.radians(elm_angle.flatten())) * r_coil, elm_z.flatten()])
CoilDefinition[1]['thin_wire_nodes_stop'] = np.column_stack([np.cos(np.radians(elm_angle_shift.flatten())) * r_coil, np.sin(np.radians(elm_angle_shift.flatten())) * r_coil, elm_z.flatten()])

CoilDefinition[1]['num_elements'] = elm_angle.shape

# Define Shielding Surface
arc_angle = 360 / (segments_angular_shield)
elm_angle_shield, elm_z = np.meshgrid(np.arange(0, segments_angular_shield) * arc_angle, np.arange(-half_length, half_length + len_step, len_step))
CoilDefinition[2]['num_elements'] = elm_angle_shield.shape
elm_angle_shift = np.roll(elm_angle_shield, -1, axis=0)

CoilDefinition[2]['thin_wire_nodes_start'] = np.column_stack([np.cos(np.radians(elm_angle.flatten())) * r_shield, np.sin(np.radians(elm_angle.flatten())) * r_shield, elm_z.flatten()])
CoilDefinition[2]['thin_wire_nodes_stop'] = np.column_stack([np.cos(np.radians(elm_angle_shift.flatten())) * r_shield, np.sin(np.radians(elm_angle_shift.flatten())) * r_shield, elm_z.flatten()])
CoilDefinition[2]['num_elements'] = elm_angle_shield.shape

# Some additional definitions for 3D contour plots
CoilDefinition[1]['Radius'] = r_coil
CoilDefinition[2]['Radius'] = r_shield

CoilDefinition[1]['Length'] = half_length * 2
CoilDefinition[2]['Length'] = half_length * 2


if plot_all == 1:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for np in range(1, 3):
        for n in range(len(CoilDefinition[np]['thin_wire_nodes_start'])):
            ax.plot([CoilDefinition[np]['thin_wire_nodes_start'][n, 0], CoilDefinition[np]['thin_wire_nodes_stop'][n, 0]], 
                    [CoilDefinition[np]['thin_wire_nodes_start'][n, 1], CoilDefinition[np]['thin_wire_nodes_stop'][n, 1]], 
                    [CoilDefinition[np]['thin_wire_nodes_start'][n, 2], CoilDefinition[np]['thin_wire_nodes_stop'][n, 2]])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Thin-wire current elements')
    ax.view_init(elev=30, azim=45)
    plt.show()


# Definition of target points in a 3D-volume
import Make_Target as mt  # Assuming you have the Make_Target module

# Define main target
TargetDefinition = {}
TargetDefinition['shape'] = 'sphere'
TargetDefinition['radius'] = 0.15
TargetDefinition['resol_radial'] = 2
TargetDefinition['resol_angular'] = 24
TargetDefinition['strength'] = 5e-3
TargetDefinition['direction'] = 'y'

target_main = mt.Make_Target(TargetDefinition)

# Possibility to plot main target
if plot_all == 1:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(target_main['points']['x1'], target_main['points']['x2'], target_main['points']['x3'], s=25, c=target_main['field'])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('Main Target Points and Field')
    ax.view_init(elev=30, azim=45)
    plt.show()

TargetDefinition['shape'] = 'cylinder'
TargetDefinition['radius'] = 0.65
TargetDefinition['length'] = 1.2
TargetDefinition['resol_radial'] = 1
TargetDefinition['resol_angular'] = 48
TargetDefinition['resol_length'] = 24
TargetDefinition['strength'] = 0e-3
TargetDefinition['direction'] = 'y'

target_shield = mt.Make_Target(TargetDefinition)


# Optionally plot shield target
if plot_all == 1:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(target_shield['points']['x1'], target_shield['points']['x2'], target_shield['points']['x3'], s=25, c=target_shield['field'])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('Shielding Target Points and Field')
    ax.view_init(elev=30, azim=45)
    plt.show()

x1 = np.concatenate((target_main['points']['x1'], target_shield['points']['x1']))
x2 = np.concatenate((target_main['points']['x2'], target_shield['points']['x2']))
x3 = np.concatenate((target_main['points']['x3'], target_shield['points']['x3']))

Points = np.column_stack([x1, x2, x3])
Target = {}
Target['Points'] = Points
num_points = len(Points)
Target['num_points'] = num_points

kn = len(x1)**2
kp = len(x1)

num_points_main = len(target_main['points']['x1'])
num_points_shield = len(target_shield['points']['x1'])


#Calculate Sensitivity
CoilDefinition[0]['StreamDirection'] = 2
CoilDefinition[1]['StreamDirection'] = 2

Sensitivity = ThinWireSensitivity(CoilDefinition, Target)


#    Calculate the unregularized Solution:
E_Mat = np.concatenate((Sensitivity[0]['ElementFieldsStream'], Sensitivity[1]['ElementFieldsStream']), axis=1)
btarget = np.concatenate((target_main['field'], target_shield['field']))
ElementCurrents_unreg = np.linalg.pinv(E_Mat) @ btarget


#    Plot unregularized current distribution:
main_stop = CoilDefinition[0]['num_elements'][0] * (CoilDefinition[0]['num_elements'][1] - 1)
ElementCurrents = {}
ElementCurrents[0] = {}
ElementCurrents[1] = {}
ElementCurrents[0]['Stream'] = np.reshape(ElementCurrents_unreg[:main_stop], Elm_angle.shape[0]-[0, 1])
ElementCurrents[1]['Stream'] = np.reshape(ElementCurrents_unreg[main_stop:], Elm_angle.shape[0]-[0, 1])

if plot_all == 1:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ElementCurrents[0]['Stream'])
    plt.colorbar()
    plt.title('a) main layer without regularization')

    plt.subplot(1, 2, 2)
    plt.imshow(ElementCurrents[1]['Stream'])
    plt.colorbar()
    plt.title('b) shielding layer without regularization')

    plt.show()


#Calculate the regularized Solution:
E_Mat = np.concatenate((Sensitivity[0]['ElementFieldsStream'], Sensitivity[1]['ElementFieldsStream']), axis=1)
btarget = np.concatenate((target_main['field'], target_shield['field']))

W = np.eye(E_Mat.shape[1])
w = W - np.roll(W, segments_angular, axis=1)
w[:, -2 * segments_angular:] = 0
z = np.zeros_like(w)
reg_mat = np.block([[w, z], [z, w]])

ElementCurrents_Reg_Weigh = TikhonovReg_Weigh(E_Mat, btarget, 5e-2, reg_mat)

#%% Add additional constraints to enforce peripheral elements to be 0
E_Mat = np.concatenate((Sensitivity[0]['ElementFieldsStream'], Sensitivity[1]['ElementFieldsStream']), axis=1)
btarget = np.concatenate((target_main['field'], target_shield['field']))

lambda_val = 1e1
W = np.eye(E_Mat.shape[1])
w = W - np.roll(W, segments_angular, axis=1)
w[:, -2 * segments_angular:] = 0

w_ext = np.block([
    [w, lambda_val * np.eye(segments_angular), np.zeros((segments_angular, E_Mat.shape[1] - segments_angular))],
    [np.zeros((segments_angular, E_Mat.shape[1] - segments_angular)), lambda_val * np.eye(segments_angular), w]
])

w_full = np.block([[w_ext, np.zeros_like(w_ext)], [np.zeros_like(w_ext), w_ext]])

ElementCurrents_Reg_Weigh = TikhonovReg_Weigh(E_Mat, btarget, 5e-1, w_full)

#%% Plot currents in 2D
import matplotlib.pyplot as plt

n_cont = 15

main_stop = CoilDefinition[0]['num_elements'][0] * (CoilDefinition[0]['num_elements'][1] - 1)

ElementCurrents_Reg_Weigh = np.array(ElementCurrents_Reg_Weigh)

ElementCurrents = [
    ElementCurrents_Reg_Weigh[:main_stop, :].reshape(elm_angle.shape[0], elm_angle.shape[1] - 1),
    ElementCurrents_Reg_Weigh[main_stop:, :].reshape(elm_angle.shape[0], elm_angle.shape[1] - 1)
]

cont_max = np.max(ElementCurrents[0]) * 1  # 0.98
cont_min = np.min(ElementCurrents[0]) * 1  # *0.98

for nP in range(2):
    plt.figure()
    plt.imshow(ElementCurrents[nP], cmap='hot', aspect='auto')
    plt.colorbar()
    plt.contour(ElementCurrents[nP], levels=np.linspace(cont_min, cont_max, n_cont), colors='k', linewidths=2)
    plt.title(f"{'a) regularized main layer' if nP == 0 else 'b) regularized shielding layer'}")

plt.show()

#%% Plot multi layer contours
import matplotlib.pyplot as plt

nP = 1
ElmtsPlot = ElementCurrents[nP - 1].reshape(CoilDefinition[nP - 1]['num_elements'][0], CoilDefinition[nP - 1]['num_elements'][1] - 1)
ElmtsPlot = np.vstack((ElmtsPlot[-1, :], ElmtsPlot, ElmtsPlot[0, :]))

cont_max_main = np.max(np.abs(ElmtsPlot))
n_cont = 15
contour_levels = np.linspace(-cont_max_main, cont_max_main, n_cont)
plt.figure()
plt.imshow(ElmtsPlot.T, cmap='hot', aspect='auto')
plt.colorbar()
plt.contour(ElmtsPlot.T, levels=contour_levels, colors='k', linewidths=2)
plt.title('Regularized main layer')

nP = 2
ElmtsPlot = ElementCurrents[nP - 1].reshape(CoilDefinition[nP - 1]['num_elements'][0], CoilDefinition[nP - 1]['num_elements'][1] - 1)
ElmtsPlot = np.vstack((ElmtsPlot[-1, :], ElmtsPlot, ElmtsPlot[0, :]))

cont_max_main = np.max(np.abs(ElmtsPlot))
n_cont = 13
contour_levels = np.linspace(-cont_max_main, cont_max_main, n_cont)
plt.figure()
plt.imshow(ElmtsPlot.T, cmap='hot', aspect='auto')
plt.colorbar()
plt.contour(ElmtsPlot.T, levels=contour_levels, colors='k', linewidths=2)
plt.title('Regularized shielding layer')

plt.show()


#%% 3D Plot of the contours
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.title('3D Coil')
ax = plt.gca()
ax.set_box_aspect([1, 1, 1])  # Set equal aspect ratio for all axes

nP = 1
S = C1.allsegs[0]  # Extract the segments from the contour data
for seg in S:
    sx = CoilDefinition[nP - 1]['Radius'] * np.cos(seg[:, 0] / (CoilDefinition[nP - 1]['num_elements'][0]) * 2 * np.pi)
    sy = CoilDefinition[nP - 1]['Radius'] * np.sin(seg[:, 0] / (CoilDefinition[nP - 1]['num_elements'][0]) * 2 * np.pi)
    sz = seg[:, 1] / len(ElmtsPlot[0, :]) * CoilDefinition[nP - 1]['Length'] - CoilDefinition[nP - 1]['Length'] / 2
    ax.plot3D(sx, sy, sz, 'b', linewidth=1)

nP = 2
S = C2.allsegs[0]  # Extract the segments from the contour data
for seg in S:
    sx = CoilDefinition[nP - 1]['Radius'] * np.cos(seg[:, 0] / (CoilDefinition[nP - 1]['num_elements'][0]) * 2 * np.pi)
    sy = CoilDefinition[nP - 1]['Radius'] * np.sin(seg[:, 0] / (CoilDefinition[nP - 1]['num_elements'][0]) * 2 * np.pi)
    sz = seg[:, 1] / len(ElmtsPlot[0, :]) * CoilDefinition[nP - 1]['Length'] - CoilDefinition[nP - 1]['Length'] / 2
    ax.plot3D(sx, sy, sz, 'r', linewidth=1)

ax.view_init(elev=-97, azim=25)
ax.axis('tight')
ax.axis('equal')
ax.axis('off')
font_size = 12
ax.tick_params(labelsize=font_size)
plt.show()

