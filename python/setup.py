from setuptools import setup, Extension
import os

openmm_dir = '@OPENMM_DIR@'
interaxplugin_header_dir = '@INTERAXPLUGIN_HEADER_DIR@'
interaxplugin_library_dir = '@INTERAXPLUGIN_LIBRARY_DIR@'
interaxplugin_install_library_dir = '@INTERAXPLUGIN_INSTALL_LIBRARY_DIR@'

extension = Extension(name='_interaxplugin',
                      sources=['InteraxPluginWrapper.cpp'],
                      libraries=['OpenMM', 'InteraxPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), interaxplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), interaxplugin_library_dir],
                      runtime_library_dirs=[os.path.join(openmm_dir, 'lib'), interaxplugin_install_library_dir],
                      extra_compile_args=['-std=c++11'],
                     )

setup(name='interaxplugin',
      version='@INTERAX_PLUGIN_VERSION@',
      py_modules=['interaxplugin', 'interaxplugin_convert', 'interaxplugin_forcefield'],
      ext_modules=[extension],
     )
