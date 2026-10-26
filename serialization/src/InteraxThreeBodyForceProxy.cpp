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

#include "InteraxThreeBodyForceProxy.h"
#include "InteraxThreeBodyForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/serialization/SerializationNode.h"
#include <sstream>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

InteraxThreeBodyForceProxy::InteraxThreeBodyForceProxy() : SerializationProxy("InteraxThreeBodyForce") {
}

void InteraxThreeBodyForceProxy::serialize(const void* object, SerializationNode& node) const {
    const InteraxThreeBodyForce& force = *reinterpret_cast<const InteraxThreeBodyForce*>(object);
    // Version 3 adds the centroid damping radius. Without centroid damping version 2 is written, so that older builds
    // still read the force (and refuse a damped one).
    bool centroidDamping = (force.getCentroidDampingRadius() != 0.0);
    node.setIntProperty("version", centroidDamping ? 3 : 2);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());
    node.setDoubleProperty("cutoff", force.getCutoffDistance());
    node.setDoubleProperty("scaleFactor", force.getScaleFactor());
    node.setBoolProperty("usesPeriodic", force.usesPeriodicBoundaryConditions());
    node.setStringProperty("alchemicalParameter", force.getAlchemicalParameterName());
    node.setIntProperty("dampingMethod", (int) force.getDampingMethod());
    node.setDoubleProperty("dampingScale", force.getDampingScale());
    if (centroidDamping)
        node.setDoubleProperty("centroidDampingRadius", force.getCentroidDampingRadius());
    SerializationNode& sites = node.createChildNode("Sites");
    for (int i = 0; i < force.getNumSites(); i++) {
        vector<int> particles;
        double alpha, frequency, width;
        int molecule;
        bool isAlchemical;
        force.getSiteParameters(i, particles, alpha, frequency, width, molecule, isAlchemical);
        SerializationNode& site = sites.createChildNode("Site");
        site.setDoubleProperty("alpha", alpha).setDoubleProperty("frequency", frequency).setDoubleProperty("width", width);
        site.setIntProperty("molecule", molecule).setBoolProperty("alchemical", isAlchemical);
        for (int particle : particles)
            site.createChildNode("Particle").setIntProperty("index", particle);
    }
}

void* InteraxThreeBodyForceProxy::deserialize(const SerializationNode& node) const {
    int version = node.getIntProperty("version");
    if (version < 1 || version > 3)
        throw OpenMMException("Unsupported version number");
    InteraxThreeBodyForce* force = new InteraxThreeBodyForce();
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));
        force->setCutoffDistance(node.getDoubleProperty("cutoff"));
        force->setScaleFactor(node.getDoubleProperty("scaleFactor"));
        force->setUsesPeriodicBoundaryConditions(node.getBoolProperty("usesPeriodic"));
        force->setAlchemicalParameterName(node.getStringProperty("alchemicalParameter"));
        if (version == 1) {
            // Version 1 had one site per molecule, without damping.
            for (auto& molecule : node.getChildNode("Molecules").getChildren()) {
                vector<int> particles;
                for (auto& particle : molecule.getChildren())
                    particles.push_back(particle.getIntProperty("index"));
                force->addSite(particles, molecule.getDoubleProperty("alpha"), molecule.getDoubleProperty("frequency"), 0.0, -1,
                        molecule.getBoolProperty("alchemical"));
            }
            return force;
        }
        force->setDampingMethod((InteraxThreeBodyForce::DampingMethod) node.getIntProperty("dampingMethod"));
        force->setDampingScale(node.getDoubleProperty("dampingScale"));
        if (version >= 3)
            force->setCentroidDampingRadius(node.getDoubleProperty("centroidDampingRadius"));
        const SerializationNode& sites = node.getChildNode("Sites");
        for (auto& site : sites.getChildren()) {
            vector<int> particles;
            for (auto& particle : site.getChildren())
                particles.push_back(particle.getIntProperty("index"));
            force->addSite(particles, site.getDoubleProperty("alpha"), site.getDoubleProperty("frequency"), site.getDoubleProperty("width"),
                    site.getIntProperty("molecule"), site.getBoolProperty("alchemical"));
        }
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
