from setuptools import setup, find_packages

setup(
   name='asterofit',
   description='An asteroseismic grid modelling analysis toolkit',
   author='Yaguang Li',
   author_email='yaguangl@hawaii.edu',
   packages=['asterofit'],  #same as name
   package_data={
		'asterofit' : ['src/*'],
   },
   install_requires=['wheel', 'scipy', 'numpy', 'pandas', 'matplotlib'], #external packages as dependencies
   extras_require={
      'test': ['pytest'],
   },
)
