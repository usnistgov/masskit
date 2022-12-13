from rdkit import Chem
import base64

try:
    import py3Dmol

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
except ModuleNotFoundError:
    pass


def file_download_button(filename='download.txt', data="", button_text="Download File",
                         css_class="p-Widget jupyter-widgets jupyter-button widget-button mod-warning"):
    """
    create an html file download button for use in jupyter notebook using HTML(file_download())
    :param filename: name of file to download
    :param data: string containing data to download
    :param button_text: the text on the button
    :param css_class: styling of the button
    :return: string containing button html
    """
    b64 = base64.b64encode(data.encode())
    payload = b64.decode()

    # BUTTONS
    html_buttons = '''<html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body>
        <a download="{filename}" href="data:text/plain;base64,{payload}" download>
        <button class="{css_class}">{button_text}</button>
        </a>
        </body>
        </html>
        '''

    html_button = html_buttons.format(payload=payload, filename=filename, css_class=css_class, button_text=button_text)
    return html_button
