import requests
import urllib


def call_airi_service(smile_in, server='10.208.85.87', port='80'):
    """
    call the airi service

    :param smile_in: the smiles string used to compute the ri value
    :param server: the http server containing the airi service
    :param port: the port on the server
    :return: the json response from the server
    """
    smile_in = urllib.parse.quote(smile_in, safe='')
    url = f'http://{server}:{port}/molecule/smiles/{smile_in}/ri/json'
    with requests.get(url) as response_in:
        return response_in.json()
