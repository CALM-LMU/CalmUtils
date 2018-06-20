from setuptools import setup

setup(
    name='CalmUtils',
    version='0.0.6',
    description='commonly used Python code at the Center for Advanced Light Micsroscopy',
    long_description=open('README.rst').read(),
    author='David Hoerl',
    author_email='hoerlatbiodotlmudotde',
    license='#TODO',
    packages=['calmutils.imageio',
              'calmutils.localization',
              'calmutils.localization.util',
              'calmutils.localization.multiview',
              'calmutils.simulation',
              'calmutils.misc',
              'calmutils.morphology'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'Pillow',
        'javabridge',
        'python-bioformats'
    ],
    package_dir={'' : './src'}
)
