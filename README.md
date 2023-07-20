# ThinWire_MRIGradientCoilDesign_python
A python fork of the matlab based "ThinWire_MRIGradientCoilDesign" gradient coil design tool for Magnetic resonance imaging (MRI)

This code demonstrates the use of thin wires to approximate a current
density. In gradient coil design for MRI usually only the z-component
of the magnetic field (Bz) is considered. Hence, only wires orthogonal to
z are used in this simulation. The Biot-Savart law is used to caclulte a 
sensitivity matrix, which is then used to calculate a current 
distribution. A regularization and an additional constraint is deployed to 
derive a ralizable coil design.
This method may be used for simple geometries. However, no generalization
for arbitrary surfaces (yet).

The code is written in Python 3, relying on standard python libaries such as numpy and matplotlib. 

An explanatory introduction is given in ThinWire_Demo.py. Two basic coils
are described in the scripts Cylindrical_SingleLayer.py and 
CylindricalShielded.py.

A publication by Sebastian Littin et al. describing the basic ideas of the matlab version of this code and gradient coil design in general can be found here: https://www.frontiersin.org/articles/10.3389/fphy.2021.699468/full

The translation is still a work in progress and has several bugs. Any bug report and help with the translation is welcome.

Niklas Wehkamp
