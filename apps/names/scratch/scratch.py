# this is just a scratch pad for various ideas

import dask.dataframe as dd
dask_name2inchi = dd.from_pandas(name2inchi, npartitions=16)

conda install fastparquet
conda install dask
dd.to_parquet(dask_name2inchi, "dask_name2inchi.pqt")



conda install pytables
import tables


store = pd.HDFStore("name2inchi.h5")
# make sure all entries in the following columns are strings
columns = ['name', 'inchi', 'inchi-key']
name2inchi.loc[:, columns] = name2inchi[columns].applymap(str)
store["name2inchi"] = name2inchi


cas2inchi.query('name == "50-78-2"')


