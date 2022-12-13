import pandas as pd
import csv

# reads in files from the pubchem ftp site and joins them together


def main():
    synonyms = pd.read_csv("CID-Synonym-filtered", sep='\t', header=None, names=["cid","name"])
    inchi = pd.read_csv("CID-InChI-Key", sep='\t', header=None, names=["cid","inchi","inchi-key"])
    name2inchi = pd.merge(synonyms, inchi, on="cid")
    name2inchi.query('name == "50-78-2"')

    name2inchi.to_csv("name2inchi.csv")
    name2inchi.to_csv("name2inchi.tsv", sep="\t", quoting=csv.QUOTE_NONE )
    name2inchi.to_pickle("name2inchi.pkl")

    cas2inchi = name2inchi[name2inchi.name.str.contains("^[1-9]{1}[0-9]{1,5}-\d{2}-\d$", na=False)]
    cas2inchi.to_csv("cas2inchi.csv")
    cas2inchi.to_csv("cas2inchi.tsv", sep="\t", quoting=csv.QUOTE_NONE )
    cas2inchi.to_pickle("cas2inchi.pkl")


if __name__ == "__main__":
    main()