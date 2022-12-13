# import libraries
from rdkit import Chem
import argparse


def main():
    # set up the arguments to the program
    parser = argparse.ArgumentParser(description='For each molecule in the query_file,'
                                                 ' check to see if molecules in the compare_file are substructures.')
    parser.add_argument('--query_file', help="molfile containing query molecules", default="")
    parser.add_argument('--compare_file', help="molfile containing molecules to compare to queries", default="")
    args = parser.parse_args()

    if args.query_file == '' or args.compare_file == '':
        raise ValueError("please specify both query_file and compare_file")

    # load in the query molecules
    query = Chem.SDMolSupplier(args.query_file)
    for query_mol in query:
        if query_mol is None:  # skip if rdkit can't compute a molecule from the query files
            continue
        # load in the comparison molecules
        compare = Chem.SDMolSupplier(args.compare_file)
        for compare_mol in compare:
            if compare_mol is None:  # skip if rdkit can't compute a molecule from the comparison file
                continue
            if query_mol.HasSubstructMatch(compare_mol):
                print(f"comparison molecule {compare_mol.GetProp('_Name')} is a substructure of {query_mol.GetProp('_Name')}")
            else:
                print(f"comparison molecule {compare_mol.GetProp('_Name')} is a not substructure of {query_mol.GetProp('_Name')}")


if __name__ == "__main__":
    main()

