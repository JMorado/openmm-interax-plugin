/* -------------------------------------------------------------------------- *
 *                            OpenMM InteraxFF Plugin                         *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM InteraxFF plugin, built on the OpenMM molecular *
 * simulation toolkit (https://openmm.org) and its example plugin.            *
 *                                                                            *
 * Portions copyright (c) 2026 the Authors.                                   *
 * Authors: Joao Morado                                                       *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "InteraxMultipoleForceProxy.h"
#include "InteraxMultipoleForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/SerializationNode.h"

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

InteraxMultipoleForceProxy::InteraxMultipoleForceProxy() : SerializationProxy("InteraxMultipoleForce") {
}

void InteraxMultipoleForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("version", 1);
    const InteraxMultipoleForce& force = *reinterpret_cast<const InteraxMultipoleForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setIntProperty("nonbondedMethod", (int) force.getNonbondedMethod());
    node.setDoubleProperty("cutoff", force.getCutoffDistance());
    node.setDoubleProperty("ewaldErrorTolerance", force.getEwaldErrorTolerance());
    double alpha;
    int nx, ny, nz;
    force.getPMEParameters(alpha, nx, ny, nz);
    node.setDoubleProperty("pmeAlpha", alpha).setIntProperty("pmeNx", nx).setIntProperty("pmeNy", ny).setIntProperty("pmeNz", nz);
    node.setBoolProperty("coreDamping", force.getCoreDamping());
    node.setDoubleProperty("mutualDampingScale", force.getMutualDampingScale());
    node.setDoubleProperty("coreDampingScale", force.getCoreDampingScale());
    node.setStringProperty("alchemicalParameter", force.getAlchemicalParameterName());
    SerializationNode& coefficients = node.createChildNode("ExtrapolationCoefficients");
    for (double c : force.getExtrapolationCoefficients())
        coefficients.createChildNode("Coefficient").setDoubleProperty("value", c);
    SerializationNode& particles = node.createChildNode("Particles");
    for (int i = 0; i < force.getNumParticles(); i++) {
        double charge, coreCharge, a, polarizability;
        int axisType, atomZ, atomX, atomY;
        bool isAlchemical;
        vector<double> dipole, quadrupole;
        force.getParticleParameters(i, charge, dipole, quadrupole, coreCharge, a, polarizability, axisType, atomZ, atomX, atomY, isAlchemical);
        SerializationNode& particle = particles.createChildNode("Particle");
        particle.setDoubleProperty("charge", charge).setDoubleProperty("coreCharge", coreCharge).setDoubleProperty("alpha", a);
        particle.setDoubleProperty("polarizability", polarizability).setIntProperty("axisType", axisType);
        particle.setIntProperty("atomZ", atomZ).setIntProperty("atomX", atomX).setIntProperty("atomY", atomY).setBoolProperty("alchemical", isAlchemical);
        SerializationNode& d = particle.createChildNode("Dipole");
        d.setDoubleProperty("d0", dipole[0]).setDoubleProperty("d1", dipole[1]).setDoubleProperty("d2", dipole[2]);
        SerializationNode& q = particle.createChildNode("Quadrupole");
        for (int k = 0; k < 9; k++)
            q.setDoubleProperty("q"+to_string(k), quadrupole[k]);
    }
    SerializationNode& exceptions = node.createChildNode("Exceptions");
    for (int i = 0; i < force.getNumExceptions(); i++) {
        int p1, p2;
        double mm, dm, dd;
        force.getExceptionParameters(i, p1, p2, mm, dm, dd);
        exceptions.createChildNode("Exception").setIntProperty("p1", p1).setIntProperty("p2", p2).setDoubleProperty("mm", mm)
                .setDoubleProperty("dm", dm).setDoubleProperty("dd", dd);
    }
}

void* InteraxMultipoleForceProxy::deserialize(const SerializationNode& node) const {
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");
    InteraxMultipoleForce* force = new InteraxMultipoleForce();
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setNonbondedMethod((InteraxMultipoleForce::NonbondedMethod) node.getIntProperty("nonbondedMethod"));
        force->setCutoffDistance(node.getDoubleProperty("cutoff"));
        force->setEwaldErrorTolerance(node.getDoubleProperty("ewaldErrorTolerance"));
        force->setPMEParameters(node.getDoubleProperty("pmeAlpha"), node.getIntProperty("pmeNx"), node.getIntProperty("pmeNy"), node.getIntProperty("pmeNz"));
        force->setCoreDamping(node.getBoolProperty("coreDamping"));
        force->setMutualDampingScale(node.getDoubleProperty("mutualDampingScale"));
        force->setCoreDampingScale(node.getDoubleProperty("coreDampingScale", 1.0));
        force->setAlchemicalParameterName(node.getStringProperty("alchemicalParameter"));
        vector<double> coefficients;
        for (auto& c : node.getChildNode("ExtrapolationCoefficients").getChildren())
            coefficients.push_back(c.getDoubleProperty("value"));
        force->setExtrapolationCoefficients(coefficients);
        for (auto& particle : node.getChildNode("Particles").getChildren()) {
            const SerializationNode& d = particle.getChildNode("Dipole");
            vector<double> dipole = {d.getDoubleProperty("d0"), d.getDoubleProperty("d1"), d.getDoubleProperty("d2")};
            const SerializationNode& q = particle.getChildNode("Quadrupole");
            vector<double> quadrupole(9);
            for (int k = 0; k < 9; k++)
                quadrupole[k] = q.getDoubleProperty("q"+to_string(k));
            force->addParticle(particle.getDoubleProperty("charge"), dipole, quadrupole, particle.getDoubleProperty("coreCharge"),
                    particle.getDoubleProperty("alpha"), particle.getDoubleProperty("polarizability"), particle.getIntProperty("axisType"),
                    particle.getIntProperty("atomZ"), particle.getIntProperty("atomX"), particle.getIntProperty("atomY"),
                    particle.getBoolProperty("alchemical"));
        }
        for (auto& e : node.getChildNode("Exceptions").getChildren())
            force->addException(e.getIntProperty("p1"), e.getIntProperty("p2"), e.getDoubleProperty("mm"), e.getDoubleProperty("dm"), e.getDoubleProperty("dd"));
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
