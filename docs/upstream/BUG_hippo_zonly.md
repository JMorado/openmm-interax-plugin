# HippoNonbondedForce (Reference/CPU): wrong forces on particles with a ZOnly frame (out-of-bounds read of `particleData[-1]`)

## Summary

On the Reference and CPU platforms, `HippoNonbondedForce` computes wrong forces for particles with the `ZOnly` axis
type. The torque-to-force mapping reads the position of the frame's x particle. For `ZOnly` that index is `-1`, so the
code reads `particleData[-1]`, which is out of bounds. The forces then depend on whatever memory lies before the
particle array:
- They can be off by hundreds of kJ/mol/nm.
- They are not reproducible: they change from run to run and between Reference and CPU.

Energies and induced dipoles are not affected; only the forces from the torques on `ZOnly` particles (and their z-axis
particles) are.

The same bug existed in `AmoebaMultipoleForce` and was fixed there in #4769 ("Fixed error with ZOnly axis type when x
particle is not specified"). The HIPPO code was not changed.

## Versions

- Observed with OpenMM 8.1.2 (conda-forge), Linux x86_64.
- The code is unchanged on `master` (checked against commit 74912095, 2025-11-11).
- The Common (CUDA/OpenCL) kernels are not affected: `mapTorqueToForce` in `hippoMultipoles.cc` builds the auxiliary
  axis from the lab x or y axis for `ZOnly`.

## Location

`plugins/amoeba/platforms/reference/src/SimTKReference/AmoebaReferenceHippoNonbondedForce.cpp`

`mapTorqueToForce()` passes the x particle unconditionally (line 1304 on master):

```cpp
mapTorqueToForceForParticle(p,
                            particleData[p.multipoleAtomZ], particleData[p.multipoleAtomX],   // multipoleAtomX == -1 for ZOnly
                            p.multipoleAtomY > -1 ? &particleData[p.multipoleAtomY] : NULL,
                            p.axisType, torques[ii], forces);
```

`mapTorqueToForceForParticle()` then always uses its position:

```cpp
Vec3 vectorV = particleI.position - particleV.position;
norms[V] = normalizeVec3(vectorV);
```

The `ZOnly` branch uses `vectorV` through `vectorUV`, `angles[UV]`, `dphi[V]` and `vectorW = vectorU.cross(vectorV)`.
The rotation to the lab frame (`applyRotationMatrixToParticle()`) is fine: for `ZOnly` it builds the x axis from the
lab x or y axis and never touches the x particle.

## Reproduction

`hippo_zonly_repro.py` (attached, OpenMM only) builds two two-particle molecules. The first particle of each has a
`ZOnly` frame (z axis particle = its partner, x particle = -1), a dipole and a cylindrically symmetric quadrupole. The
script compares the analytic forces with central finite differences of the energy. It creates a few other Contexts
first, which changes what lies in memory before the particle data.

```
$ python hippo_zonly_repro.py Reference
OpenMM 8.1.2, Reference platform
particle 0: analytic [-254.56907  -49.21733  282.67887]  finite difference [-235.49217  -35.33945    6.92351]
particle 1: analytic [ 3533.89101  1609.5469  -2114.85052]  finite difference [ 3514.81414  1595.669   -1839.09515]
particle 2: analytic [-3744.13662 -1513.74923  2842.93543]  finite difference [-3744.13665 -1513.74921  2842.93539]
particle 3: analytic [  464.81468   -46.58033 -1010.76378]  finite difference [  464.81468   -46.58033 -1010.76377]
max |analytic - finite difference| = 2.758e+02 kJ/mol/nm

$ python hippo_zonly_repro.py CPU
OpenMM 8.1.2, CPU platform
...
max |analytic - finite difference| = 3.565e-05 kJ/mol/nm
```

On the CPU platform the same input happened to give correct forces in this run. The result depends on the memory
contents, not on the platform.

We first saw this in a periodic box of 140 methanol molecules where the methyl carbon has a `ZOnly` frame:
- Two runs of the same configuration gave different HIPPO forces on that carbon and its O, by about 4 kJ/mol/nm.
- One of the runs did not agree with finite differences.

## Suggested fix

Do what #4769 did for `AmoebaReferenceMultipoleForce`: pass the x particle as a pointer that may be `NULL`, and for
`ZOnly` use the same lab axis as the rotation:

```cpp
// mapTorqueToForce()
mapTorqueToForceForParticle(p, particleData[p.multipoleAtomZ],
                            p.multipoleAtomX > -1 ? &particleData[p.multipoleAtomX] : NULL,
                            p.multipoleAtomY > -1 ? &particleData[p.multipoleAtomY] : NULL,
                            p.axisType, torques[ii], forces);

// mapTorqueToForceForParticle(), with particleV a pointer
Vec3 vectorV;
if (axisType != HippoNonbondedForce::ZOnly)
    vectorV = particleI.position - particleV->position;
else {
    Vec3 vectorZ = -vectorU;   // vectorU = particleI - particleU, i.e. minus the z axis
    vectorV = (fabs(vectorZ[0]) < 0.866 ? Vec3(1, 0, 0) : Vec3(0, 1, 0));
}
norms[V] = normalizeVec3(vectorV);
```

For a `ZOnly` particle the mapping does not depend on which auxiliary direction is used, as long as it is not parallel
to z. We applied this change in a copy of the HIPPO reference code (our plugin):
- The reproduction's forces agree with finite differences to 4e-5 kJ/mol/nm on both platforms, which is the
  finite-difference error.
- The methanol box's forces agree to 2e-8 relative to the rms force.
