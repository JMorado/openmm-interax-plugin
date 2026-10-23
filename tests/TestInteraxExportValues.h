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

/**
 * This tests a platform against values of the InteraxFF export: the energies and forces of the InteraxFF
 * dispersion, exchange repulsion, short-range correction and three-body dispersion, computed by the export
 * (interax.openmm, Custom forces) on the Reference platform in double precision. The systems and values are in
 * tests/data, written by tests/python/generate_export_values.py: system.xml holds the plugin forces converted from
 * the export, one term per force group, and reference.txt the positions and the values of each term.
 *
 * Each platform test includes this file and calls runExportValueTests().
 */

#include "OpenMMInterax.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/State.h"
#include "openmm/System.h"
#include "openmm/VerletIntegrator.h"
#include "openmm/serialization/XmlSerializer.h"
#include "openmm/internal/AssertionUtilities.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace InteraxPlugin;
using namespace OpenMM;
using namespace std;

/**
 * The values of the export for one test system.
 */
struct ExportValues {
    bool periodic;
    Vec3 box[3];
    vector<Vec3> positions;
    vector<string> terms;
    vector<double> energies;
    vector<vector<Vec3> > forces;
};

static string getDataDirectory(const string& name) {
    return string(INTERAX_TEST_DATA_DIR)+"/"+name;
}

static ExportValues loadExportValues(const string& name) {
    ifstream file((getDataDirectory(name)+"/reference.txt").c_str());
    if (!file.is_open())
        throw OpenMMException("Cannot open the export values of "+name);
    ExportValues values;
    string line, keyword;
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        stringstream header(line);
        header >> keyword;
        if (keyword == "box") {
            string first;
            header >> first;
            values.periodic = (first != "none");
            if (values.periodic) {
                stringstream vectors(line.substr(line.find(' ')));
                for (int i = 0; i < 3; i++)
                    vectors >> values.box[i][0] >> values.box[i][1] >> values.box[i][2];
            }
        }
        else if (keyword == "positions") {
            int numParticles;
            header >> numParticles;
            values.positions.resize(numParticles);
            for (Vec3& p : values.positions)
                file >> p[0] >> p[1] >> p[2];
        }
        else if (keyword == "term") {
            string term;
            int group;
            double energy;
            header >> term >> group >> energy;
            ASSERT_EQUAL(values.terms.size(), group);
            values.terms.push_back(term);
            values.energies.push_back(energy);
            values.forces.push_back(vector<Vec3>(values.positions.size()));
            for (Vec3& f : values.forces.back())
                file >> f[0] >> f[1] >> f[2];
        }
    }
    return values;
}

static System* loadSystem(const string& name) {
    ifstream file((getDataDirectory(name)+"/system.xml").c_str());
    if (!file.is_open())
        throw OpenMMException("Cannot open the system of "+name);
    return XmlSerializer::deserialize<System>(file);
}

/**
 * Set the overlap method of the exchange repulsion and the short-range correction.
 */
static void setOverlapMethod(System& system, InteraxExchangeForce::OverlapMethod method) {
    for (int i = 0; i < system.getNumForces(); i++) {
        if (InteraxExchangeForce* force = dynamic_cast<InteraxExchangeForce*>(&system.getForce(i)))
            force->setOverlapMethod(method);
        if (InteraxShortRangeForce* force = dynamic_cast<InteraxShortRangeForce*>(&system.getForce(i)))
            force->setOverlapMethod((InteraxShortRangeForce::OverlapMethod) method);
    }
}

/**
 * Compare the energy and forces of each term with the export.
 *
 * @param name             the test system (a directory of tests/data)
 * @param exactOverlap     if true, the Slater overlaps use the Exact method instead of the InteraxFF scheme of the
 *                         export, and differ from it by the export's truncation error
 * @param energyTolerance  the tolerance of the energy of each term, relative to its magnitude
 * @param forceTolerance   the tolerance of every force component, relative to the RMS force of the term
 */
static void compareWithExport(const string& name, Platform& platform, const map<string, string>& properties, bool exactOverlap,
        double energyTolerance, double forceTolerance) {
    ExportValues values = loadExportValues(name);
    unique_ptr<System> system(loadSystem(name));
    if (exactOverlap)
        setOverlapMethod(*system, InteraxExchangeForce::Exact);
    VerletIntegrator integrator(0.001);
    Context context(*system, integrator, platform, properties);
    if (values.periodic)
        context.setPeriodicBoxVectors(values.box[0], values.box[1], values.box[2]);
    context.setPositions(values.positions);
    for (int group = 0; group < values.terms.size(); group++) {
        State state = context.getState(State::Energy | State::Forces, false, 1<<group);
        double energy = state.getPotentialEnergy();
        const vector<Vec3>& forces = state.getForces();
        double sumSquared = 0, maxError = 0;
        for (int i = 0; i < forces.size(); i++) {
            sumSquared += values.forces[group][i].dot(values.forces[group][i]);
            for (int k = 0; k < 3; k++)
                maxError = max(maxError, fabs(forces[i][k]-values.forces[group][i][k]));
        }
        double rmsForce = sqrt(sumSquared/(3*forces.size()));
        double energyError = fabs(energy-values.energies[group])/fabs(values.energies[group]);
        cout << "  " << name << " " << values.terms[group] << (exactOverlap ? " (exact overlap)" : "") << ": relative energy error " << energyError
                << ", force error " << maxError/rmsForce << " of the RMS force" << endl;
        ASSERT_EQUAL_TOL(values.energies[group], energy, energyTolerance);
        if (maxError > forceTolerance*rmsForce)
            throw OpenMMException("Forces of "+values.terms[group]+" in "+name+" differ from the export");
    }
}

/**
 * Run the tests on a platform.
 *
 * @param doublePrecision  whether the platform computes the interactions in double precision. In double precision
 *                         the plugin must reproduce the export to rounding; in single precision the InteraxFF order 2
 *                         scheme of the export loses precision (docs/SPEC.md), and only the Exact method is tested.
 */
static void runExportValueTests(Platform& platform, const map<string, string>& properties, bool doublePrecision) {
    const char* systems[] = {"cluster", "methanol_box"};
    for (const char* name : systems) {
        if (doublePrecision) {
            compareWithExport(name, platform, properties, false, 1e-8, 1e-7);
            compareWithExport(name, platform, properties, true, 1e-6, 1e-5);
        }
        else
            compareWithExport(name, platform, properties, true, 2e-6, 5e-5);
    }
}
