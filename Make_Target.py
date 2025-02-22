import numpy as np

def Make_Target(TargetDefinition):
    B_out = {}
    # Define different shapes
    if TargetDefinition['shape'] == 'cube':
        print('cube')
      #  r1 = TargetDefinition['length'] / 2
      #  d1 = TargetDefinition['length'] / (TargetDefinition['resolution'])# - 1)
      #  x1, x2, x3 = np.mgrid[-r1:r1:d1, -r1:r1:d1, -r1:r1:d1]
    elif TargetDefinition['shape'] == 'cuboid':
        print('cuboid')
      #  r1 = TargetDefinition['length'][0] / 2
      #  r2 = TargetDefinition['length'][1] / 2
      #  r3 = TargetDefinition['length'][2] / 2
      #  d1 = TargetDefinition['length'][0] / (TargetDefinition['resolution'][0])# - 1)
      #  d2 = TargetDefinition['length'][1] / (TargetDefinition['resolution'][1])# - 1)
      #  d3 = TargetDefinition['length'][2] / (TargetDefinition['resolution'][2])# - 1)
      #  x1, x2, x3 = np.mgrid[-r1:r1:d1, -r2:r2:d2, -r3:r3:d3]
    elif TargetDefinition['shape'] == 'slice':
        print('slice')
      #  r1 = TargetDefinition['length'][0] / 2
      #  r2 = TargetDefinition['length'][1] / 2
      #  r3 = TargetDefinition['length'][2] / 2
      #  d1 = TargetDefinition['length'][0] / (TargetDefinition['resolution'][0])# - 1)
      #  d2 = TargetDefinition['length'][1] / (TargetDefinition['resolution'][1])# - 1)
      #  d3 = TargetDefinition['length'][2] / (TargetDefinition['resolution'][2])# - 1)
      #  if d1 == 0:
      #      x1, x2, x3 = np.mgrid[0, -r2:r2:d2, -r3:r3:d3]
      #  elif d2 == 0:
      #      x1, x2, x3 = np.mgrid[-r1:r1:d1, 0, -r3:r3:d3]
      #  elif d3 == 0:
      #      x1, x2, x3 = np.mgrid[-r1:r1:d1, -r2:r2:d2, 0]
      #  else:
      #      x1, x2, x3 = np.mgrid[-r1:r1:d1, -r2:r2:d2, -r3:r3:d3]
    elif TargetDefinition['shape'] == 'sphere':
        print('sphere')
        r = TargetDefinition['radius']
        print("r",r)
        d1 = TargetDefinition['radius'] / (TargetDefinition['resol_radial'] - 1)
        print("d1",d1)
        d2 = np.pi / (TargetDefinition['resol_angular'] - 1) 
        print("d2",d2)
        d3 = 2 * np.pi / (TargetDefinition['resol_angular'] - 1)
        print("d3",d3)
        #print("r+d1",(r+d1))
        print("TargetDefinition['resol_radial']*d1",(TargetDefinition['resol_radial']-1)*d1)
        ra, theta, phi = np.mgrid[0:(r+d1/2):d1, 0:(np.pi+d2/2):d2, (-np.pi):(np.pi+d3/2):d3] #!!!Attention mgrid is not inclusive vs matlab ndgrid is inclusive
        #ra, theta, phi = np.mgrid[d1:(r+d1):d1, 0:(np.pi+d2):d2, (-np.pi):(np.pi+d3):d3] #!!!Attention mgrid is not inclusive vs matlab ndgrid is inclusive
        #ra, theta, phi = np.mgrid[0:(r+d1):d1, 0:(np.pi+d2):d2, -np.pi:(np.pi+d3):d3]
        print('ra.shape',ra.shape)
        #print('theta',theta)
        #print('phi',phi)
        x1 = ra * np.sin(theta) * np.cos(phi)
        print('x1',x1.shape)
        x2 = ra * np.sin(theta) * np.sin(phi)
        print('x2',x2.shape)
        x3 = ra * np.cos(theta)
        print('x3',x3.shape)
#    elif TargetDefinition['shape'] == 'sphere':
#        print('sphere')
#        r = TargetDefinition['radius']
#        print("r",r)
#        d1 = TargetDefinition['radius'] / (TargetDefinition['resol_radial'])# - 1)
#        print("d1",d1)
#        d2 = np.pi / (TargetDefinition['resol_angular'])# - 1)
#        print("d2",d2)
#        d3 = 2 * np.pi / (TargetDefinition['resol_angular'])# - 1)
#        print("d3",d3)
#        ra, theta, phi = np.mgrid[0:r:d1, 0:np.pi:d2, -np.pi:np.pi:d3]
#        print('ra',ra)
#        print('theta',theta)
#        print('phi',phi)
#        x1 = ra * np.sin(theta) * np.cos(phi)
#        x2 = ra * np.sin(theta) * np.sin(phi)
#        x3 = ra * np.cos(theta)
    elif TargetDefinition['shape'] == 'cylinder':
        print('cylinder')
      #  r = TargetDefinition['radius']
      #  d2 = 2 * np.pi / (TargetDefinition['resol_angular'])# - 1)
      #  print("d2",d2)
      #  d3 = TargetDefinition['length'] / (TargetDefinition['resol_length'])# - 1)
      #  print("d3",d3)
      #  l1 = TargetDefinition['length'] / 2
      #  if TargetDefinition['resol_radial'] == 1:
      #      ra, phi, z = np.mgrid[r, 0:2 * np.pi:d2, -l1:l1:d3]
      #  else:
      #      d1 = TargetDefinition['radius'] / (TargetDefinition['resol_radial'])# - 1)
      #      ra, phi, z = np.mgrid[0:r:d1, 0:2 * np.pi:d2, -l1:l1:d3]
      #  x1 = ra * np.cos(phi)
      #  x2 = ra * np.sin(phi)
      #  x3 = z

    B_out['points'] = {'x1': x1.ravel(), 'x2': x2.ravel(), 'x3': x3.ravel()}
    B_field = np.zeros_like(x1)

    # Define field strength
    if TargetDefinition['direction'] == 'x':
        B_field = x1.ravel() / np.max(x1.ravel()) * TargetDefinition['strength']
    elif TargetDefinition['direction'] == 'y':
        B_field = x2.ravel() / np.max(x2.ravel()) * TargetDefinition['strength']
    elif TargetDefinition['direction'] == 'z':
        B_field = x3.ravel() / np.max(x3.ravel()) * TargetDefinition['strength']
    else:
        B_field = x1.ravel() / np.max(x1.ravel()) * 1e-3

    B_out['field'] = B_field

    return B_out

