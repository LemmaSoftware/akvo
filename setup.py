#!/usr/bin/env python
import sys

# optionally use qt4
# if sys.argv[1] == "build_ui":
#     try:
#         from pyqt_distutils.build_ui import build_ui
#         cmdclass = {'build_ui': build_ui}
#     except ImportError:
#         build_ui = None  # user won't have pyqt_distutils when deploying
#         cmdclass = {}
# else:
#     build_ui = None  # user won't have pyqt_distutils when deploying
#     cmdclass = {}

# Require PyQt5 and compiltion of GUI files via pyuic 
from setuptools import setup, Extension
from setuptools.command.build_py import build_py
from pyqt_distutils.build_ui import build_ui

class custom_build_py(build_py):
    def run(self):
        self.run_command('build_ui')
        build_py.run(self)

try:
    from Cython.Build import cythonize as cythonise
except ImportError:
    def cythonise(*args, **kwargs):
        #from Cython.Build import cythonize
        #return cythonize(*args, **kwargs)
        return 

#from distutils.core import setup

setup(name='Akvo',
      version='1.0.5',
      description='Surface nuclear magnetic resonance workbench',
      author='Trevor P. Irons',
      author_email='Trevor.Irons@lemmasoftware.org',
      url='https://svn.lemmasofware.org/akvo',
      #setup_requires=['PyQt5'],
      setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'PyQt5','setuptools>=18.0',
      ],
#      ext_modules = cythonise("akvo/tressel/*.pyx"), 
#      build_requires=['cython'],
      install_requires=[
#          'cython',
          'rpy2',
          'matplotlib',
          'scipy',
          'numpy',
          'PyQt5',
          'pyyaml',
          'pyqt-distutils',
          'cmocean'
      ],
      packages=['akvo', 'akvo.tressel', 'akvo.gui'],
      license=['GPL 4.0'],
      entry_points = {
              'console_scripts': [
                  'akvo = akvo.gui.akvoGUI:main',                  
              ],              
          },
      # for forced build of pyuic
      cmdclass={
          'build_ui': build_ui,
          'build_py': custom_build_py,
      },
      #cmdclass=cmdclass,
      # Mechanism to include auxiliary files
      include_package_data=True,
      package_data={
        'akvo.gui': ['*.png']  #All .r files 
      },
    )


