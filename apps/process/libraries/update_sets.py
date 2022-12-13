import logging
import hydra
from omegaconf import DictConfig
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

"""
Takes the set field in one parquet file and updates the set of the corresponding records in another parquet file.
Correspondence is done by inchi key molecular connectivity (ignores stereochemistry).

Edit the conf/config_update_sets.yaml file, then run this python script.
python update_sets.py 
"""


@hydra.main(config_path="conf", config_name="config_update_sets")
def update_sets_app(config: DictConfig) -> None:

    logging.getLogger().setLevel(logging.INFO)

    print(f"File with new set info: {config.input_source.file.name}")
    print(f"File to be updated: {config.input_update.file.name}")
    
    source_table = pq.read_table(config.input_source.file.name, columns=['inchi_key', 'set'])
    source_table = source_table.append_column("connectivity",
                                              pc.utf8_slice_codeunits(source_table['inchi_key'],0,14))

    update_table = pq.read_table(config.input_update.file.name, columns=['inchi_key', 'set'])
    update_table = update_table.append_column("connectivity",
                                              pc.utf8_slice_codeunits(update_table['inchi_key'],0,14))

    connectivity_dict = dict()
    for i in range(source_table.num_rows):
        connectivity_dict[source_table['connectivity'][i]] = source_table['set'][i].as_py()
        #print(source_table['connectivity'][i], source_table['set'][i])

    new_set = []
    for i in range(update_table.num_rows):
        query = update_table['connectivity'][i]
        if query in connectivity_dict:
            new_set.append(connectivity_dict[query])
        else:
            new_set.append(update_table['set'][i].as_py())

    set_array = pa.array(new_set, type=pa.dictionary(pa.int8(), pa.string()))
    print(f"Updated rows: {update_table.num_rows}")

    print("Reading full dataset")
    update_table = pq.read_table(config.input_update.file.name)
    idx = update_table.column_names.index('set')
    print(f"Replacing column: {idx}")
    update_table = update_table.set_column(idx, 'set', set_array)
    print(f"Writing new dataset to {config.output.file.name}")
    pq.write_table(update_table, config.output.file.name, row_group_size=5000)
    
if __name__ == "__main__":
    update_sets_app()
