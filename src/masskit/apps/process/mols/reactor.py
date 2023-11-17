from contextlib import contextmanager
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import logging
import hydra
from omegaconf import DictConfig
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from rdkit import Chem
from masskit.data_specs.arrow_types import MolArrowType
from masskit.small_molecule.react import Reactor
from masskit.utils.general import read_arrow, write_arrow

"""
takes a parquet file with rdkit mols in it and expands the file by doing reactions and tautomerization
"""


# def mol2path(mol, max_path_length=5):
#     return jsonpickle.encode(get_shortest_paths(
#             mol, max_path_length), keys=True)


@hydra.main(config_path="conf", config_name="config_reactor", version_base=None)
def reactor_app(config: DictConfig) -> None:

    logging.getLogger().setLevel(logging.INFO)

    input_file = Path(config.input.file.name).expanduser()
    table = read_arrow(input_file)

    # check that id field exists and is unique
    assert pa.compute.count_distinct(table['id']).as_py() == len(table)

    mols = table['mol'].combine_chunks().to_numpy()
    ids = table['id'].combine_chunks().to_numpy()

    # new products
    products = []
    # original ids
    orig_id = []

    reactor = Reactor()

    # run reactor over entire set of mols, keeping a parallel list of orig_id
    for i, mol in enumerate(mols):
        new_products = reactor.react(mol, 
                                    reactant_names=config.conversion.reactant_names,
                                    functional_group_names=config.conversion.functional_group_names,
                                    num_tautomers=config.conversion.num_tautomers,
                                    max_products=config.conversion.max_products,
                                    mass_range=config.conversion.mass_range,
                                    max_passes=config.conversion.max_passes,
                                    include_original_molecules=config.conversion.include_original_molecules,
                                    )
        products.extend([Chem.rdMolInterchange.MolToJSON(x) for x in new_products])
        orig_id.extend([ids[i]]*len(new_products))
    
    # create table using new mols and orig_id and new ids (which are just sequential)
    new_table = pa.Table.from_arrays([pa.array(np.arange(len(products)), type=pa.uint64()), 
                                      pa.array(orig_id, type=pa.uint64())], names=['id', 'orig_id'])
    new_table = new_table.append_column('mol', pa.array(products, type=MolArrowType()))

    table = table.rename_columns([x if x != 'id' else 'orig_id' for x in table.column_names])
    # do a join with original table (delete mol before)
    table = table.drop_columns('mol')
    new_table = new_table.join(table, 'orig_id')

    # with Pool(config.get('num_workers', 8)) as p:
    #     shortest_paths = p.map(partial(mol2path, 
    #                                    max_path_length=config.conversion.max_path_length), 
    #                                    mols)

    # output the files
    output_file = Path(config.output.file.name).expanduser()
    write_arrow(new_table, output_file)

if __name__ == "__main__":
    reactor_app()
