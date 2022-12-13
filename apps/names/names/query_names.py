# load the adapter
import psycopg2
import argparse

# load the psycopg extras module
import psycopg2.extras

"""
given a list of names, return corresponding inchikeys and inchis
Lewis Geer
"""


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--server', help="database server", default="astrocyte0.campus.nist.gov")
    parser.add_argument('--name', nargs='+',
                        help='space delimited list of names.  use double quotes around a name if it has spaces',
                        required=True)
    parser.add_argument('--dbname', help="database name",  default="small_mol")
    parser.add_argument('--user', help="username", default="readonly")
    parser.add_argument('--password', help="password", default="batchnorm")

    args = parser.parse_args()

    # Try to connect
    with psycopg2.connect(f"host='{args.server}' dbname='{args.dbname}' user='{args.user}' password='{args.password}'")\
            as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as curs:
            curs.execute(f"select * from name2inchi(ARRAY{args.name})")
            rows = curs.fetchall()
            for row in rows:
                print(f"{row['name']}\t{row['inchikey']}\t{row['inchi']}")


if __name__ == "__main__":
    main()
