from setuptools import setup

setup(
    name='tandem',
    version='1.0',
    description='tools for tandem analysis',
    author='Lewis Geer',
    author_email='lewis.geer@nist.gov',
    packages=['tandem'],
    install_requires=[],
    entry_points={
        'console_scripts': [
            'check_substructure = tandem.check_substructure:main',
        ]
    }
)
