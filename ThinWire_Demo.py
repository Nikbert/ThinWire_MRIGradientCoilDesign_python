import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Make_Target as Make_Target
import ThinWireSensitivity as tws

# coil description: Cylindrical unshielded coil

# define coil-parameters of a current matrix with wire elements orthogonal to z

CoilDefinition = {}
CoilDefinition['Partitions'] = 1
segments_angular = 16 #56
half_length = 0.6  # 600mm
len_step = 0.12 #0.02  # 20mm
r_coil = 0.4  # 800mm coil diameter

arc_angle = 360 / segments_angular
elm_angle, elm_z = np.meshgrid(np.arange(0, segments_angular) * arc_angle, np.arange(-half_length, half_length + len_step, len_step))
#elm_angle, elm_z = np.meshgrid(np.arange(0, segments_angular) * arc_angle, np.arange(-half_length, half_length + len_step, len_step))
print('elm_angle', elm_angle )
CoilDefinition[0] = {}  # Initialize CoilDefinition[1] as an empty dictionary
#CoilDefinition[1] = {}  # Initialize CoilDefinition[1] as an empty dictionary
CoilDefinition[0]['num_elements'] = elm_angle.shape

elm_angle_shift = np.roll(elm_angle, -1, axis=1)
print('elm_angle_shift', elm_angle_shift )
#elm_angle_shift = np.roll(elm_angle, -1, axis=0)


# Define Cylindrical Main Surface
CoilDefinition[0]['thin_wire_nodes_start'] = np.column_stack([np.cos(np.radians(elm_angle.flatten())) * r_coil, np.sin(np.radians(elm_angle.flatten())) * r_coil, elm_z.flatten()])

CoilDefinition[0]['thin_wire_nodes_stop'] = np.column_stack([np.cos(np.radians(elm_angle_shift.flatten())) * r_coil, np.sin(np.radians(elm_angle_shift.flatten())) * r_coil, elm_z.flatten()])

CoilDefinition[0]['num_elements'] = elm_angle.shape

# Definition of target points in a 3D-volume

TargetDefinition = {}
TargetDefinition['shape'] = 'sphere'
TargetDefinition['radius'] = 0.2
TargetDefinition['resol_radial'] = 3
TargetDefinition['resol_angular'] = 15
TargetDefinition['strength'] = 5e-3
TargetDefinition['direction'] = 'y'

target_points = Make_Target.Make_Target(TargetDefinition)

# plot target

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(target_points['points']['x1'], target_points['points']['x2'], target_points['points']['x3'], s=25, c=target_points['field'], cmap='jet')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('Target Points and Field')
ax.view_init(elev=30, azim=45)
#plt.show()

x1 = target_points['points']['x1']
x2 = target_points['points']['x2']
x3 = target_points['points']['x3']

Points = np.column_stack([x1, x2, x3])

Target = {}
Target['Points'] = Points
num_points = len(Points)
Target['num_points'] = num_points

kn = len(x1) ** 2
kp = len(x1)


# plot the thin wire elements

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print("len(CoilDefinition[0]['thin_wire_nodes_start'])",len(CoilDefinition[0]['thin_wire_nodes_start'][0][:]))
print("(CoilDefinition[0]['thin_wire_nodes_start'])",CoilDefinition[0]['thin_wire_nodes_start'][:][1])
print("(CoilDefinition[0]['thin_wire_nodes_stop'])",CoilDefinition[0]['thin_wire_nodes_stop'][:][1])
#plt.plot(CoilDefinition[0]['thin_wire_nodes_start'][:][0], CoilDefinition[0]['thin_wire_nodes_stop'][:][1]) #, 
for n in range(len(CoilDefinition[0]['thin_wire_nodes_start'])):
    #ax.plot(CoilDefinition[0]['thin_wire_nodes_start'][n][0], CoilDefinition[0]['thin_wire_nodes_stop'][n][0]) #, 
    plt.plot([CoilDefinition[0]['thin_wire_nodes_start'][n][0], CoilDefinition[0]['thin_wire_nodes_stop'][n][0]], 
            [CoilDefinition[0]['thin_wire_nodes_start'][n][1], CoilDefinition[0]['thin_wire_nodes_stop'][n][1]], 
            [CoilDefinition[0]['thin_wire_nodes_start'][n][2], CoilDefinition[0]['thin_wire_nodes_stop'][n][2]])

#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
ax.set_title('Thin-wire current elements')
ax.view_init(elev=30, azim=45)
plt.show()

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#
#print("len(CoilDefinition[0]['thin_wire_nodes_start'])",CoilDefinition[0]['thin_wire_nodes_start'])
#for n in range(len(CoilDefinition[0]['thin_wire_nodes_start'])):
#    ax.plot([CoilDefinition[0]['thin_wire_nodes_start'][n][0], CoilDefinition[0]['thin_wire_nodes_stop'][n][0]]), 
#        [CoilDefinition[0]['thin_wire_nodes_start'][n][1], CoilDefinition[0]['thin_wire_nodes_stop'][n][1]], 
#        [CoilDefinition[0]['thin_wire_nodes_start'][n][2], CoilDefinition[0]['thin_wire_nodes_stop'][n][2]])
#
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#ax.set_title('Thin-wire current elements')
#ax.view_init(elev=30, azim=45)
#plt.show()

# Some definitions for 3D contour plotting...
CoilDefinition[0]['Radius'] = r_coil
CoilDefinition[0]['Length'] = half_length * 2


# Calculate the sensitivity matrix
Sensitivity = tws.ThinWireSensitivity(CoilDefinition, Target)

btarget = target_points['field']
ElementCurrents = np.linalg.pinv(Sensitivity['ElementFields']) @ btarget
ResultingField = Sensitivity['ElementFields'] @ ElementCurrents

# plot the unregularized current distribution
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(ElementCurrents.reshape(elm_angle.shape), aspect='auto', cmap='jet')
ax.set_title('Unregularized current distribution')
ax.set_xlabel('circumferential [rad]')
ax.set_ylabel('z-Axis [m]')
fig.colorbar(im)
plt.show()


ElementCurrents = TikhonovReg(Sensitivity['ElementFields'], btarget, 0.0077)

# plot the Tikhonov-regularized current distribution
fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(ElementCurrents.reshape(elm_angle.shape), aspect='auto', cmap='jet')
ax.set_title('Regularized current distribution')
ax.set_xlabel('circumferential [rad]')
ax.set_ylabel('z-Axis [m]')
fig.colorbar(im)
plt.show()


ElementCurrents_Balance_reshape = ElementCurrents.reshape(elm_angle.shape)

Stream_Reg = np.cumsum(ElementCurrents_Balance_reshape[:,::-1], axis=1)
Stream_Reg_rev = np.cumsum(ElementCurrents_Balance_reshape, axis=1)

Stream = np.zeros(Stream_Reg.shape[0], Stream_Reg.shape[1]+1)
Stream[:,1:] = Stream_Reg/2
Stream[:,:-1] = Stream[:,:-1] - Stream_Reg_rev[:,::-1]/2

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(Stream, aspect='auto', cmap='jet')
cont_max = np.max(Stream)*0.98
n_cont = 15
cont = ax.contour(Stream.T, levels=np.arange(-cont_max, cont_max, 2*cont_max/n_cont), colors='k', linewidths=2)
ax.set_title('Stream function from integrating the currents')
ax.set_xlabel('circumferential [rad]')
ax.set_ylabel('z-Axis [m]')
fig.colorbar(im)
plt.show()


eye_ang = np.eye(elm_angle.shape[0])
ElementFields_Add3D = np.repeat(eye_ang[:, :, np.newaxis], elm_angle.shape[1], axis=2)
ElementFields_Add = ElementFields_Add3D.reshape(-1, elm_angle.shape[1])
TargetFields_Add = np.zeros(elm_angle.shape[0])

ElementFields_Balance = np.vstack((Sensitivity.ElementFields, ElementFields_Add*5e-4))
TargetField_Balance = np.concatenate((btarget, TargetFields_Add))

ElementCurrents_Balance = TikhonovReg(ElementFields_Balance, TargetField_Balance, 0.00005)

ResultingField_Balance = Sensitivity.ElementFields @ ElementCurrents_Balance

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(ElementCurrents_Balance.reshape(elm_angle.shape), aspect='auto', cmap='jet')
ax.set_title('Regularized currents with balance')
ax.set_xlabel('circumferential [rad]')
ax.set_ylabel('z-Axis [m]')
fig.colorbar(im)
plt.show()

ElementCurrents_Balance_reshape = ElementCurrents_Balance.reshape(elm_angle.shape)

Stream_Reg = np.cumsum(ElementCurrents_Balance_reshape[:,::-1], axis=1)
Stream_Reg_rev = np.cumsum(ElementCurrents_Balance_reshape, axis=1)

Stream = np.zeros((Stream_Reg.shape[0], Stream_Reg.shape[1]+1))
Stream[:,1:] = Stream_Reg/2
Stream[:,:-1] = Stream[:,:-1] - Stream_Reg_rev[:,::-1]/2

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(Stream, aspect='auto', cmap='jet')
ax.set_title('Stream function from integrating the currents')
ax.set_xlabel('circumferential [rad]')
ax.set_ylabel('z-Axis [m]')
fig.colorbar(im)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(Stream, aspect='auto', cmap='jet')
ax.set_title('Stream function from integrating the currents')
ax.set_xlabel('circumferential [rad]')
ax.set_ylabel('z-Axis [m]')
cont_max = np.max(Stream) * 0.98
n_cont = 15
contours = ax.contour(Stream.T, levels=np.linspace(-cont_max, cont_max, n_cont), colors='k', linewidths=2)
ax.clabel(contours, inline=True, fontsize=8)
fig.colorbar(im)
plt.show()

#%% 3D Plot of the stream function
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(sx_ph, sy_ph, sz_ph, rstride=1, cstride=1, facecolors=plt.cm.jet(Stream_p), shade=False)
ax.set_xlabel('x-Axis [m]')
ax.set_ylabel('y-Axis [m]')
ax.set_zlabel('z-Axis [m]')
ax.view_init(elev=-7, azim=25)
fig.colorbar(surf)
plt.title('Stream function in 3D representation')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(S)):
    sxp = r_coil * np.cos(S[i].allsegs[0][:, 0] / (CoilDefinition[nP].num_elements[1]) * 2 * np.pi)
    syp = r_coil * np.sin(S[i].allsegs[0][:, 0] / (CoilDefinition[nP].num_elements[1]) * 2 * np.pi)
    szp = S[i].allsegs[0][:, 1] / len(Stream[0, :]) * CoilDefinition[nP].Length - CoilDefinition[nP].Length / 2
    
    ax.plot(sxp, syp, szp, 'b', linewidth=1)

ax.set_xlabel('x-Axis [m]')
ax.set_ylabel('y-Axis [m]')
ax.set_zlabel('z-Axis [m]')
ax.view_init(elev=-7, azim=25)
plt.title('3D contours')
plt.axis('tight')
plt.axis('equal')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(S)):
    sxp = r_coil * np.cos(S[i].allsegs[0][:, 0] / (CoilDefinition[nP].num_elements[1]) * 2 * np.pi)
    syp = r_coil * np.sin(S[i].allsegs[0][:, 0] / (CoilDefinition[nP].num_elements[1]) * 2 * np.pi)
    szp = S[i].allsegs[0][:, 1] / len(Stream[0, :]) * CoilDefinition[nP].Length - CoilDefinition[nP].Length / 2
    
    ax.plot(sxp, syp, szp, 'k', linewidth=2)

sx_ph = np.vstack((sx_ph, sx_ph[0, :]))
sy_ph = np.vstack((sy_ph, sy_ph[0, :]))
sz_ph = np.vstack((sz_ph, sz_ph[0, :]))
Stream_p = np.vstack((Stream_p, Stream_p[0, :]))

ax.plot_surface(sx_ph, sy_ph, sz_ph, facecolors=plt.cm.viridis(Stream_p / np.max(Stream_p)), edgecolor='none')

ax.set_xlabel('x-Axis [m]')
ax.set_ylabel('y-Axis [m]')
ax.set_zlabel('z-Axis [m]')
ax.view_init(elev=-7, azim=25)
plt.title('3D coil')
plt.axis('tight')
plt.axis('equal')
plt.axis('off')
plt.show()

