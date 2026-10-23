"""
Generate the reference values of the plugin tests (tests/data) from the InteraxFF export.

For each test system, the InteraxFF export (interax.openmm.create_system_from_topology, model E1) is evaluated on the
Reference platform in double precision, term by term, and the values are written next to the converted plugin
forces:

  tests/data/<system>/system.xml      a System with only the four plugin forces, converted from the export with the
                                      export's overlap method (convert_system(overlap="export")); force group k holds
                                      the k'th term of TERMS
  tests/data/<system>/reference.txt   the positions, the box, and the energy and forces of each term in the export

The tests (tests/TestInteraxExportValues.h on every platform, tests/python/test_export_values.py) load these files
and need neither InteraxFF nor its model. Regenerate them when the model or the export changes:

  python generate_export_values.py     (with the interax environment; see compare_export.py)
"""
import datetime
import os
import subprocess

from compare_export import (COMPONENTS, INTERAX_REPO, MODEL, box_topology, cluster_topology, evaluate, interaxplugin_convert,
                            load_model, openmm, terms_by_kind)
from interax.openmm import create_system_from_topology

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
TERMS = ["dispersion", "exchange", "short_range", "three_body"]
SYSTEMS = {
    "cluster": (cluster_topology, False, "six different molecules (H/C/N/O/S/Cl), non-periodic, no cutoff"),
    "methanol_box": (box_topology, True, "140 methanol, periodic, 0.9 nm cutoff, switching from 0.8 nm, dispersion LRC"),
}


def plugin_system(export):
    """The plugin forces of a converted export, alone in a new System, with force group k for the k'th term."""
    converted = interaxplugin_convert.convert_system(export, overlap="export")
    indices = terms_by_kind(converted, True)
    system = openmm.System()
    for i in range(converted.getNumParticles()):
        system.addParticle(converted.getParticleMass(i))
    system.setDefaultPeriodicBoxVectors(*converted.getDefaultPeriodicBoxVectors())
    for group, kind in enumerate(TERMS):
        force = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(converted.getForce(indices[kind])))
        force.setForceGroup(group)
        system.addForce(force)
    return system


def main():
    ff, nagl = load_model()
    commit = subprocess.run(["git", "-C", INTERAX_REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    for name, (make_topology, periodic, description) in SYSTEMS.items():
        topology, positions = make_topology()
        export = create_system_from_topology(topology, ff, nagl, electrostatics="hippo", nonbonded_cutoff=0.9, components=COMPONENTS)
        box = export.getDefaultPeriodicBoxVectors() if periodic else None
        values = evaluate(export, positions, box, "Reference", terms_by_kind(export, False))
        system = plugin_system(export)
        directory = os.path.join(DATA, name)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "system.xml"), "w") as file:
            file.write(openmm.XmlSerializer.serialize(system))
        with open(os.path.join(directory, "reference.txt"), "w") as file:
            file.write(f"# InteraxFF export values: {description}\n")
            file.write(f"# model {os.path.basename(MODEL)}, InteraxFF {commit}, OpenMM {openmm.__version__}, Reference platform, double "
                       f"precision, generated {datetime.date.today()} by tests/python/generate_export_values.py\n")
            file.write("# Units: nm, kJ/mol, kJ/mol/nm. Each term is force group k of system.xml, in the order listed.\n")
            n = system.getNumParticles()
            if periodic:
                vectors = [v.value_in_unit(openmm.unit.nanometer) for v in box]
                file.write("box " + " ".join(f"{x:.17g}" for v in vectors for x in v) + "\n")
            else:
                file.write("box none\n")
            file.write(f"positions {n}\n")
            for p in positions:
                file.write(" ".join(f"{x:.17g}" for x in p) + "\n")
            for group, kind in enumerate(TERMS):
                energy, forces = values[kind]
                file.write(f"term {kind} {group} {energy:.17g}\n")
                for f in forces:
                    file.write(" ".join(f"{x:.17g}" for x in f) + "\n")
        print(f"{name}: {n} particles, " + ", ".join(f"{kind} {values[kind][0]:.6f}" for kind in TERMS))


if __name__ == "__main__":
    main()
