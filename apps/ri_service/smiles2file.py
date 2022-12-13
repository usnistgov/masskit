import pandas as pd
import os
from sys import exit
from rdkit import Chem
from masskit.small_molecule import threed
import json
import argparse
import logging
logger = logging.getLogger()

def standardize_smiles(smile_in):
  """
  convert smiles string to a mol and standardize it
  then output the canonical isomeric smiles
  :param smile_in: input smiles
  :return: canonical smiles, mol
  """
  mol = Chem.MolFromSmiles(smile_in)
  if mol is None:
    logger.info(f'Unable to parse {smile_in}')
    return None, None
  try:
    mol = threed.standardize_mol(mol)
  except ValueError as e:
    # log here
    logger.info(f'Unable to standardize {smile_in}')
    return None, None
  canonical = Chem.MolToSmiles(mol)
  return canonical, mol


def canonicalize_smiles(input_smiles):
  """
  calculate the ri
  :param input: the input smiles as an array
  :return: the output as a dict, canonical smiles -> input smiles
  """
  output = {}
  for smile in input_smiles:
    # canonicalize smiles
    canonical, mol = standardize_smiles(smile)
    if mol is None:
      logger.info(f'Unable to parse {smile}')
      continue
    output[canonical] = smile
  return output


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='convert smiles into standardized smiles')
  parser.add_argument('--input', default="", help='input file, one smiles per line')
  parser.add_argument('--map_output', default="", help='output file, json map from canonical smiles to input smiles')
  parser.add_argument('--standardized_output', default="", help='output file, list of standardized smiles')
  args = parser.parse_args()
  
  with open(args.input) as file:
    input_smiles = file.readlines()
    input_smiles = [s.rstrip() for s in input_smiles]
    output = canonicalize_smiles(input_smiles)
    if not output:
      exit(2)
    # write output to map file as json
    with open(args.map_output, 'w') as file:
      file.write(json.dumps(output))
    # write output to raw csv file
    with open(args.standardized_output, 'w') as file:
      for key, value in output.items():
        file.write(f'{key},1\n')
    exit(0)
  exit(1)  

