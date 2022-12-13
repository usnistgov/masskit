import subprocess
from subprocess import Popen, PIPE
from subprocess import check_output
from flask import Flask
import tempfile
import os
from urllib.parse import unquote_plus
import sys
import json

app = Flask(__name__)

"""
to run:
export PYTHONPATH=/mnt/c/Users/lewis/source/deep/spectra/ei/PA-Graph-Transformer:/mnt/c/Users/lewis/source/deep/spectra/ei/nistms2
export FLASK_APP=ri_service.py:app
export FLASK_ENV=development
flask run

if the app detects that it is "frozen" by pyinstaller, then the program looks in ./{shortest_paths,split_data,train_prop}
for executables.  Otherwise the executables used are in $CWD/PA-Graph-Transformer/{parse,preprocess,train}
"""

def run_command(command):
    session = subprocess.Popen(command, stdout=PIPE, stderr=PIPE) # , env={'PYTHONPATH':os.environ['PYTHONPATH']}
    stdout, stderr = session.communicate()
    if stderr:
        raise Exception("Error "+str(stderr))
    return stdout.decode('utf-8')


@app.route("/molecule/smiles/<smiles>/ri/<format>")
def calc_ri(smiles, format):
  """
  calculate the ri
  :param smiles: the input smiles (may be comma delimited)
  :param format: the format of the output
  """
  smiles = unquote_plus(smiles)
  smiles = smiles.split(",")
  if not smiles:
    return {'error': f'no smiles string specified'}, 400
  if format.lower() != 'json':
    return {'error': f'unknown file format {format}'}, 400
  tmpdirname = tempfile.mkdtemp()
  with open(f"{tmpdirname}/smiles_in.csv", 'w+') as fp:
    for smile in smiles:
      # canonicalize smiles
      fp.write(f"{smile.rstrip()}\n")
  try:
    # is running in production?
    if os.environ.get('FLASK_ENV', '') != 'development':
      smiles2file_path = ['dist/smiles2file/smiles2file']
      split_data_path = ['dist/smiles2file/split_data']
      shortest_paths_path = ['dist/smiles2file/shortest_paths']
      train_prop_path = ['dist/smiles2file/train_prop']
      model_path = '.'
    else:
      cwd = os.path.dirname(os.path.abspath(__file__))
      smiles2file_path = ['python', f'{cwd}/smiles2file.py']
      split_data_path = ['python', f'{cwd}/PA-Graph-Transformer/parse/split_data.py']
      shortest_paths_path = ['python', f'{cwd}/PA-Graph-Transformer/preprocess/shortest_paths.py']
      train_prop_path = ['python', f'{cwd}/PA-Graph-Transformer/train/train_prop.py']
      model_path = cwd
    subprocess.run([*smiles2file_path, '--input', f'{tmpdirname}/smiles_in.csv', '--map_output', f'{tmpdirname}/map.json', '--standardized_output', f'{tmpdirname}/raw.csv'], check=True, env=os.environ)
    subprocess.run([*split_data_path, '-data_path', f'{tmpdirname}/raw.csv', '-splits', '0,0,1', '-output_dir', f'{tmpdirname}'], check=True, env=os.environ)
    subprocess.run([*shortest_paths_path, '-data_dir', f'{tmpdirname}'], env=os.environ)
    subprocess.run([*train_prop_path, '-test_mode', '-test_model', f'{model_path}/model_99', '-data', f'{tmpdirname}', '-loss_type', 'mae', '-output_dir', f'{tmpdirname}', '-model_type', 'transformer', '-hidden_size', '160', '-p_embed', '-ring_embed', '-max_path_length', '3', '-lr', '5e-4', '-no_share', '-n_heads', '1', '-d_k', '160', '-dropout', '0.0'], env=os.environ)
  except Exception as e:
    app.logger.info(f'command line failure: {e}')
    return {'error': f'unable to compute ri for {smiles}'}, 500
  # read in map
  with open(f"{tmpdirname}/map.json") as fp:
    try:
      smiles_map = json.load(fp)
      # read in csv
      output = {}
      with open(f"{tmpdirname}/test.txt") as fp:
        line = fp.readline()
        while line:
          line = line.rstrip()
          values = line.split(',')
          output[smiles_map[values[0]]] = float(values[2])*6280.0
          line = fp.readline()
        return output
    except Exception as e:
      app.logger.info(f'command line failure: {e}')
      return {'error': f'unable to map ri values'}, 500
  
  # fix ri_service.ini so that it uses this file,
  # modify pyinstaller to install smiles2file, but not this file
  # don't forget modification on aws server 10-208-85-10 spec file: pathex=['.', './nistms2'], for smiles2file.py

if __name__ == "__main__":
    app.run(host='0.0.0.0')
