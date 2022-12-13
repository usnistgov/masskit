from setuptools import setup

setup(
    name='names',
    version='1.0',
    description='query postgres db that contains chemical naming info',
    author='Lewis Geer',
    author_email='lewis.geer@nist.gov',
    packages=['names'],
    install_requires=['psycopg2', 'pandas'],
    entry_points={
        'console_scripts': [
            'query_names = names.query_names:main',
            'import_pubchem = names.import_pubchem:main',
        ]
    }
)


# import sys
# from cx_Freeze import setup, Executable
#
# build_exe_options = {"packages": ["psycopg2", "psycopg2._psycopg", "pandas"]}
#
# setup(
#    name='names',
#    version='1.0',
#    description='query postgres db that contains chemical naming info',
#    author='Lewis Geer',
#    author_email='lewis.geer@nist.gov',
#    packages=['names'],
#    options={"build_exe": build_exe_options, 'bdist_msi': {'add_to_path': True}},
#    executables=[Executable("scripts/query_names.py", base=None)]
# )
#
# # to build, python setup.py bdist_msi
# # currently builds an installer, but install is missing psycopg2