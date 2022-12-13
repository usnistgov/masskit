# Installation
1. install a python distribution if not already on your computer, preferably via [anaconda](https://www.anaconda.com/distribution/#download-section)
2. install the rdkit library `conda install -c conda-forge rdkit`
3. in a command window, run `pip install "git+https://gitlab.nist.gov/gitlab/msdc/msdc_services.git#egg=tandem&subdirectory=apps/tandem"`

# Running
`check_substructure --query_file query.sdf --compare_file compare.sdf`

