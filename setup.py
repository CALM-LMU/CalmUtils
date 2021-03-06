from setuptools import setup

setup(
    name='CalmUtils',
    version='0.0.13-dev2',
    description='commonly used Python code at the Center for Advanced Light Micsroscopy',
    long_description=open('README.rst').read(),
    author='David Hoerl',
    author_email='hoerlatbiodotlmudotde',
    license='#TODO',
    packages=['calmutils',
              'calmutils.descriptors',
              'calmutils.filter',
              'calmutils.imageio',
              'calmutils.localization',
              'calmutils.localization.util',
              'calmutils.localization.multiview',
              'calmutils.simulation',
              'calmutils.misc',
              'calmutils.misc.json',
              'calmutils.morphology',
              'calmutils.stitching',
              'calmutils.stitching.fusion'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'Pillow',
    ],
    package_dir={'' : './src'}
)
