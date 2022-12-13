import py3Dmol
from rdkit import Chem


def draw_3d(m, p=None, conf_id=-1):
    """
    draw a conformer in 3d using pymol.  Note that you need to "pip install py3Dmol" and if
    you want to use in jupyter notebook, install "jupyter labextension install jupyterlab_3dmol"
    :param m: input molecule
    :param p: py3Dmol view
    :param conf_id: conformer id in m to draw
    :return: image
    """
    mb = Chem.MolToMolBlock(m, confId=conf_id)
    if p is None:
        p = py3Dmol.view(width=400, height=400)
    p.removeAllModels()
    p.addModel(mb, 'sdf')
    p.setStyle({'stick': {}})
    p.setBackgroundColor('0xeeeeee')
    p.zoomTo()
    return p.show()
