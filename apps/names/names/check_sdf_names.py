# load the adapter
import psycopg2
import argparse
import pandas as pd
import re
import logging

# load the psycopg extras module
import psycopg2.extras

"""
given a list of names, return corresponding inchikeys and inchis.  Classify results by type of match
Lewis Geer
"""


def unescape_string(string_in):
    """
    turn strings like '.alpha.' into 'alpha'
    :param string_in: input string
    :return: fixed string
    """
    return re.sub(r"\.(\w+)\.", r"\1", string_in)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--server', help="database server", default="astrocyte0.campus.nist.gov")
    parser.add_argument('--input', help='name of the pkl file containing the dataframe to check',
                        required=True)
    parser.add_argument('--output', help='output csv')
    parser.add_argument('--log', help='name of log file')
    parser.add_argument('--dbname', help="database name",  default="small_mol")
    parser.add_argument('--user', help="username", default="readonly")
    parser.add_argument('--password', help="password", default="batchnorm")

    args = parser.parse_args()

    if args.log:
        logging.basicConfig(filename=args.log)
        logging.getLogger().setLevel(logging.INFO)

    # Try to connect
    df = pd.read_pickle(args.input)
    with psycopg2.connect(f"host='{args.server}' dbname='{args.dbname}' user='{args.user}' password='{args.password}'")\
            as conn:
        id_col = []
        name_col = []
        inchi_key_col = []
        status_col = []
        status_val_col = []
        for chemical in df.itertuples():
            names = [chemical.name, *chemical.synonyms]
            names = [x for x in names if not x.startswith('$')]  # get rid of synonyms that are actually notes
            names = [x for x in names if not x.endswith('#')]  # get rid of synonyms that are guesses
            names = [unescape_string(x) for x in names]   # unescape names
            df_names = pd.read_sql_query("select * from name2inchi(%(names)s)", conn, params={"names": names})
            df_inchi = pd.read_sql_query("select * from inchi2name(%(inchi_key)s)", conn,
                                         params={"inchi_key": chemical.inchi_key})
            df_concat = pd.concat([df_names, df_inchi])
            df_csv = df_concat.to_csv()
            # if df_concat = 0, report 0
            if len(df_concat.index) == 0:
                status = "no name or inchikey matches"
                status_val = 6
            # match names and inchi with only one unique inchi
            elif len(df_names.index) > 0 and len(df_inchi) > 0 and df_concat["inchikey"].nunique() == 1:
                status = "name matches and only one inchikey match"
                status_val = 0
            elif len(df_names.index) == 0 and df_concat["inchikey"].nunique() == 1:
                status = "inchikey match, but no name match"
                status_val = 4
            elif len(df_names.index) > 0 and len(df_inchi) == 0:
                # check for to see if all of the inchis have a connectivty match
                if df_names.inchikey.str.match(chemical.inchi_key[0:14]).all():
                    status = "name matches and inchikey matches are connectivity matches.  no full inchikey match"
                    status_val = 2
                else:
                    status = "name matches but no inchikey match, even for connectivity"
                    status_val = 3
            else:
                if df_concat.inchikey.str.match(chemical.inchi_key[0:14]).all():
                    status = "name matches and multiple inchikey matches are all at least connectivity matches"
                    status_val = 1
                else:
                    status = "mixture of name matches and matches to multiple inchikeys without connectivity matches"
                    status_val = 5

            logging.info(f"{chemical.Index}; {chemical.inchi_key}; {chemical.name}\nstatus: {status}\ndata: {df_csv}\n")

            # put id, name, isomeric_smiles, and inchi key into result
            id_col.append(chemical.Index)
            name_col.append(chemical.name)
            inchi_key_col.append(chemical.inchi_key)
            status_col.append(status)
            status_val_col.append(status_val)

        # change result into df_result and write out to csv.
        df_result = pd.DataFrame.from_dict({"id": id_col, "name": name_col, "inchi_key": inchi_key_col,
                                            "status": status_col, "status_val": status_val_col})
        df_result = df_result.sort_values(by="status_val")
        if args.output:
            df_result.to_csv(args.output)


if __name__ == "__main__":
    main()