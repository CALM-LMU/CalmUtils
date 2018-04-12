from setuptools import setup

setup(
    name='CalmUtils',
    version='0.0.1',
    description='commonly used Python code at the Center for Advanced Light Micsroscopy',
    long_description=open('README.rst').read(),
    author='David Hoerl',
    author_email='hoerlatbiodotlmudotde',
    license='#TODO',
    packages=['src.calmutils.imageio',
              'src.calmutils.localization'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-image',
        'Pillow',
        'javabridge',
        'python-bioformats'
    ]
)
