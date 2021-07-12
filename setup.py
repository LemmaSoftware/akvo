#!/usr/bin/env python

# Requires PyQt5 and compiltion of GUI files via pyuic 
from setuptools import setup, Extension
from setuptools.command.build_py import build_py

try:
    from pyqt_distutils.build_ui import build_ui
except ImportError:
    print("Please install pyqt_distutils")
    print( "(sudo) pip(3) install pyqt-distutils")
    exit()

class custom_build_py(build_py):
    def run(self):
        self.run_command('build_ui')
        build_py.run(self)


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='Akvo',     
      version='1.6.1', 
      python_requires='>3.7.0', # due to pyLemma 
      description='Surface nuclear magnetic resonance workbench',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Trevor P. Irons',
      author_email='Trevor.Irons@lemmasoftware.org',
      url='https://akvo.lemmasoftware.org/',
      #setup_requires=['PyQt5'],
      setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        #'PyQt', 
        'pyqt_distutils',
        'PyQt5',
        'setuptools>=18.0',
      ],
#      ext_modules = cythonise("akvo/tressel/*.pyx"), 
#      build_requires=['cython'],
      install_requires=[
#         'cython',
#          'rpy2',
          'matplotlib',
          'scipy',
          'padasip',
          'seaborn',
          'numpy',
          'pyqt5',
          'pyyaml',
          'ruamel.yaml',
          'pandas',
          'pyqt-distutils',
          'cmocean',
          'multiprocessing',
          'itertools',
          'pyLemma >= 0.4.0'
      ],
      packages=['akvo', 'akvo.tressel', 'akvo.gui'],
      license='GPL 4.0',
      entry_points = {
              'console_scripts': [
                  'akvo = akvo.gui.akvoGUI:main',                  
                  'akvoK0 = akvo.tressel.calcAkvoKernel:main',                  
                  'akvoQT = akvo.tressel.invertTA:main',                  
              ],              
          },
      #cmdclass = cmdclass,
      # for forced build of pyuic
      cmdclass={
          'build_ui': build_ui,
          'build_py': custom_build_py,
      },
      # Mechanism to include auxiliary files
      # commenting out may be necessary? 
      #include_package_data=True,
      package_data={
        'akvo.gui': ['*.png'], 
        'akvo.gui': ['*.ui'],
        '': ['*.png'], 
      },
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
      ],
    )

