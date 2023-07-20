# coil description: Cylindrical unshielded coil
plot_all = 1  # set to 1, to optionally plot intermediate steps

# define coil parameters of the matrix coil: segments_angular, half_length, len_step
CoilDefinition.Partitions = 1
segments_angular = 56
half_length = 0.45  # 450mm
len_step = 0.015  # 15mm
r_coil = 0.4  # 800mm coil diameter

arc_angle = 360 / segments_angular
elm_angle, elm_z = np.mgrid[0:segments_angular * arc_angle:arc_angle, -half_length:half_length:len_step]
CoilDefinition[0].num_elements = elm_angle.shape
elm_angle_shift = np.roll(elm_angle, -1, axis=0)

# Define Cylindrical Surface
CoilDefinition[0].thin_wire_nodes_start = np.column_stack(
    [np.cos(np.deg2rad(elm_angle)).ravel() * r_coil, np.sin(np.deg2rad(elm_angle)).ravel() * r_coil,
     elm_z.ravel()])
CoilDefinition[0].thin_wire_nodes_stop = np.column_stack(
    [np.cos(np.deg2rad(elm_angle_shift)).ravel() * r_coil, np.sin(np.deg2rad(elm_angle_shift)).ravel() * r_coil,
     elm_z.ravel()])
CoilDefinition[0].num_elements = elm_angle.shape

# possibility to plot thin wire elements
if plot_all == 1:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for n in range(len(CoilDefinition[0].thin_wire_nodes_start)):
        ax.plot([CoilDefinition[0].thin_wire_nodes_start[n, 0], CoilDefinition[0].thin_wire_nodes_stop[n, 0]],
                [CoilDefinition[0].thin_wire_nodes_start[n, 1], CoilDefinition[0].thin_wire_nodes_stop[n, 1]],
                [CoilDefinition[0].thin_wire_nodes_start[n, 2], CoilDefinition[0].thin_wire_nodes_stop[n, 2]])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Thin-wire current elements')
    plt.show()

# Some definitions for 3D contour plotting...
CoilDefinition[0].Radius = r_coil
CoilDefinition[0].Length = half_length * 2

# Definition of main target points in a 3D-volume
TargetDefinition = {
    'shape': 'sphere',
    'radius': 0.2,
    'resol_radial': 3,
    'resol_angular': 15,
    'resol_length': 12,
    'strength': 5e-3,
    'direction': 'y'
}
target_points = Make_Target(TargetDefinition)

# plot target
if plot_all == 1:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(target_points['points']['x1'], target_points['points']['x2'], target_points['points']['x3'],
               s=np.ones_like(target_points['points']['x1']) * 25, c=target_points['field'])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Target Points and Field')
    plt.show()

x1 = target_points['points']['x1']
x2 = target_points['points']['x2']
x3 = target_points['points']['x3']

Points = np.column_stack([x1, x2, x3])
Target = {'Points': Points, 'num_points': len(Points)}

kn = len(x1) ** 2
kp = len(x1)


# Calculate sensitivity matrix
CoilDefinition[0]['StreamDirection'] = 2
Sensitivity = ThinWireSensitivity(CoilDefinition, Target)

# Calculate an unregularized Solution
btarget = target_points['field']
ElementCurrents_Unreg = []
ElementCurrents_Unreg.append({'Stream': np.linalg.pinv(Sensitivity[0]['ElementFieldsStream'].reshape(-1, CoilDefinition[0]['num_elements'][0] * CoilDefinition[0]['num_elements'][1])) @ btarget})

# Plot the unregularized solution
if plot_all == 1:
    plt.figure()
    plt.imshow(ElementCurrents_Unreg[0]['Stream'].reshape(elm_angle.shape[0], elm_angle.shape[1] - 1), aspect='auto')
    plt.colorbar()
    plt.title('Unregularized Stream Function')
    PlotThinWireStreamFunction3D(CoilDefinition, ElementCurrents_Unreg)

# Calculate the (wrong) regularized Solution
ElementCurrents_Reg = TikhonovReg(Sensitivity[0]['ElementFieldsStream'].reshape(-1, CoilDefinition[0]['num_elements'][0] * CoilDefinition[0]['num_elements'][1]), btarget, 0.2077)

plt.figure()
plt.imshow(ElementCurrents_Reg.reshape(elm_angle.shape[0], elm_angle.shape[1] - 1), aspect='auto')
plt.colorbar()

# Use alternative regularization matrix for regularizing effective currents
E_Mat = Sensitivity[0]['ElementFieldsStream'].reshape(-1, CoilDefinition[0]['num_elements'][0] * CoilDefinition[0]['num_elements'][1])
W = np.eye(E_Mat.shape[1])
w = W - np.roll(W, segments_angular, axis=1)
w[:2 * segments_angular, -2 * segments_angular:] = np.zeros((2 * segments_angular, 2 * segments_angular))

ElementCurrents_Reg = TikhonovReg_Weigh(E_Mat, btarget, 5e-1, w)

plt.figure()
plt.imshow(ElementCurrents_Reg.reshape(elm_angle.shape[0], elm_angle.shape[1] - 1), aspect='auto')
plt.colorbar()


# Add additional constraints to enforce peripheral elements to be 0
btarget = target_points['field']
lambda_val = 1e1
E_Mat = Sensitivity[0]['ElementFieldsStream'].reshape(-1, CoilDefinition[0]['num_elements'][0] * CoilDefinition[0]['num_elements'][1])
W = np.eye(E_Mat.shape[1])
w = W - np.roll(W, segments_angular, axis=1)
w[:2 * segments_angular, -2 * segments_angular:] = np.zeros((2 * segments_angular, 2 * segments_angular))

w_ext = np.vstack((np.hstack((w, np.hstack((lambda_val * np.eye(segments_angular), np.zeros((segments_angular, E_Mat.shape[1] - segments_angular)))))),
                   np.hstack((np.hstack((np.zeros((segments_angular, E_Mat.shape[1] - segments_angular)), lambda_val * np.eye(segments_angular))), w))))

btarget_ext = np.vstack((btarget, np.zeros((CoilDefinition[0]['Partitions'] * 2, 1))))

ElementCurrents_Reg = TikhonovReg_Weigh(E_Mat, btarget_ext, 5e-1, w_ext)

plt.figure()
plt.imshow(ElementCurrents_Reg.reshape(elm_angle.shape[0], elm_angle.shape[1] - 1), aspect='auto')
plt.colorbar()


# Perform additional constraints to enforce peripheral elements to be 0
btarget = target_points['field']

red_count1 = np.arange(0, CoilDefinition[0]['num_elements'][0])
red_count2 = np.arange(CoilDefinition[0]['num_elements'][0] * (CoilDefinition[0]['num_elements'][1] - 2), CoilDefinition[0]['num_elements'][0] * (CoilDefinition[0]['num_elements'][1] - 1))
red_count = np.concatenate((red_count1, red_count2))
red_countd = np.concatenate((np.roll(red_count1, 1), np.roll(red_count2, 1)))

diff_constraint = Sensitivity[0]['ElementFieldsStream'][:, red_count] - Sensitivity[0]['ElementFieldsStream'][:, red_countd]

E_Mat = Sensitivity[0]['ElementFieldsStream'].reshape(-1, CoilDefinition[0]['num_elements'][0] * CoilDefinition[0]['num_elements'][1])

E_Mat_ext = np.hstack((E_Mat, 5e3 * diff_constraint))
btarget_ext = np.vstack((btarget, np.zeros((diff_constraint.shape[1], 1))))

n_stop = CoilDefinition[0]['num_elements'][0] * (CoilDefinition[0]['num_elements'][1] - 1)

W = np.eye(E_Mat.shape[1])

w = W - np.roll(W, segments_angular, axis=1)
w[:2 * segments_angular, -2 * segments_angular:] = np.zeros((2 * segments_angular, 2 * segments_angular))

w_ext = np.vstack((np.hstack((w, np.zeros((E_Mat.shape[1], 2 * segments_angular)))), np.hstack((np.zeros((2 * segments_angular, E_Mat.shape[1])), np.eye(2 * segments_angular)))))

ElementCurrents_temp = TikhonovReg_Weigh(E_Mat_ext, btarget, 5e-1, w_ext)

n_stop = CoilDefinition[0]['num_elements'][0] * (CoilDefinition[0]['num_elements'][1] - 1)
ElementCurrents_Reg = ElementCurrents_temp[:n_stop]

plt.figure()
plt.imshow(ElementCurrents_Reg.reshape(elm_angle.shape[0], elm_angle.shape[1] - 1), aspect='auto')
plt.colorbar()


#%% Plot currents in 2D
ElementCurrentsReg[0]['Stream'] = ElementCurrents_Reg.reshape(elm_angle.shape[0], elm_angle.shape[1] - 1)

if plot_all == 1:
    plt.figure()
    plt.imshow(ElementCurrentsReg[0]['Stream'], aspect='auto')
    plt.colorbar()
    plt.title('Regularized Stream Function')

# PlotThinWireStreamFunction3D(CoilDefinition, ElementCurrentsReg)
ContourPlotThinWireStreamFunction3D(CoilDefinition, ElementCurrentsReg, 19)

