import logging
from contextlib import contextmanager
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import hydra
import jsonpickle
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig
from rdkit import Chem

from masskit.data_specs.arrow_types import PathArrowType
from masskit.utils.general import read_arrow, write_arrow

"""
takes a parquet file with rdkit mols in it and adds shortest path information
"""

def ordered_pair(a1, a2):
    if a1 > a2:
        return (a2, a1)
    else:
        return (a1, a2)


def get_ring_paths(rd_mol):
    rings_dict = {}
    ssr = [list(x) for x in Chem.GetSymmSSSR(rd_mol)]
    for ring in ssr:
        ring_sz = len(ring)
        is_aromatic = True
        for atom_idx in ring:
            if not rd_mol.GetAtoms()[atom_idx].GetIsAromatic():
                is_aromatic = False
                break
        for ring_idx, atom_idx in enumerate(ring):
            for other_idx in ring[ring_idx:]:
                atom_pair = ordered_pair(atom_idx, other_idx)
                if atom_pair not in rings_dict:
                    rings_dict[atom_pair] = [(ring_sz, is_aromatic)]
                else:
                    if (ring_sz, is_aromatic) not in rings_dict[atom_pair]:
                        rings_dict[atom_pair].append((ring_sz, is_aromatic))
    return rings_dict


def get_shortest_paths(rd_mol, max_path_length=5):
    """Returns the shortest paths for the given rd_mol.

    For every pair of atoms, if they are connected by a path <= max_path_length
        the atoms on that path are enumerated. Otherwise, if there is a path,
        then the truncated path is given.
    The returned object is a tuple of dictionaries. The first dictionary
        contains the shortest paths for those with <= max_path_length. The
        second contains a pointer to the truncated path. The third contains
        the ring information for atoms inside rings.
    """
    fragments = Chem.rdmolops.GetMolFrags(rd_mol)

    def get_atom_frag(fragments, atom_idx):
        """Returns the fragment the atom belongs to."""
        for frag in fragments:
            if atom_idx in frag:
                return frag
        assert False  # All valid indices should belong to a fragment

    n_atoms = rd_mol.GetNumAtoms()
    paths_dict = {}  # Contains atom pairs with path length <= max_path_length
    pointer_dict = {}  # Contains atom pairs with path length > max_path_length

    for atom_idx in range(n_atoms):
        atom_frag = get_atom_frag(fragments, atom_idx)  # List of atoms in the atom's fragment

        # this iteration avoids visiting the same atom pair more than once
        for other_idx in range(atom_idx + 1, n_atoms, 1):
            if other_idx not in atom_frag:
                continue

            shortest_path = Chem.rdmolops.GetShortestPath(
                rd_mol, atom_idx, other_idx)

            path_length = len(shortest_path) - 1  # shortest_path counts atoms
            if path_length > max_path_length:
                pointer_dict[
                    (atom_idx, other_idx)] = shortest_path[max_path_length]
                pointer_dict[
                    (other_idx, atom_idx)] = shortest_path[-1 - max_path_length]
            else:
                paths_dict[(atom_idx, other_idx)] = shortest_path

    ring_dict = get_ring_paths(rd_mol)
    return paths_dict, pointer_dict, ring_dict


@contextmanager
def batch_reader(input_file, row_batch_size=5000):
        format = Path(input_file).suffix[1:].lower()
        if format == 'parquet':
            dataset = pq.ParquetFile(input_file)
        elif format == 'arrow':
            dataset = pa.ipc.RecordBatchStreamReader(pa.memory_map(input_file, 'r'))
        else:
            raise ValueError(f"incorrect file type {format}")
        
        try:
            if format == 'parquet':
                for batch in dataset.iter_batches():
                    table = pa.Table.from_batches(batch)  # schema is inferred from batch
                    yield table 
            elif format == 'arrow':
                while True:
                    batch = dataset.read_next_batch()
                    table = pa.Table.from_batches([batch])
                    yield table
        finally:
            dataset.close()

class BatchWriter:
    """
    Doesn't admit well to being a context manager as the file can't be
    opened until write_table() is called as the file open functions
    require a table schema
    """
    def __init__(self, output_file):
        self.format = Path(output_file).suffix[1:].lower()
        if self.format not in ['arrow', 'parquet']:
            raise ValueError(f"incorrect file type {self.format}")
        self.dataset = None

    def write_table(self, table):
        if self.format == 'parquet':
            if self.dataset == None:
                self.dataset = pq.ParquetWriter(pa.OSFile(self.filename, 'wb'), table.schema)
            self.dataset.write_table(table)
        elif self.format == 'arrow':
            if self.dataset == None:
                self.dataset = pa.RecordBatchFileWriter(pa.OSFile(self.filename, 'wb'), table.schema)
            self.dataset.write_table(table)

    def close(self):
        if self.dataset is not None:
            self.dataset.close()

    def __del__(self):
        # explictly close as writer can be threaded
        self.close()


def mol2path(mol, max_path_length=5):
    return jsonpickle.encode(get_shortest_paths(
            mol, max_path_length), keys=True)


@hydra.main(config_path="conf", config_name="config_path", version_base=None)
def path_generator_app(config: DictConfig) -> None:

    logging.getLogger().setLevel(logging.INFO)

    input_file = Path(config.input.file.name).expanduser()

    table = read_arrow(input_file)

    shortest_paths = []
    mols = table['mol'].combine_chunks().to_numpy()

    with Pool(config.get('num_workers', 8)) as p:
        shortest_paths = p.map(partial(mol2path, 
                                       max_path_length=config.conversion.max_path_length), 
                                       mols)

    # delete shortest_paths column if it already exists
    try:
        table = table.remove_column(table.column_names.index("shortest_paths"))
    except ValueError:
        pass
    new_arrays = []
    new_array = pa.array(shortest_paths)
    if type(new_array) is pa.ChunkedArray:
        for array in new_array.iterchunks():
            new_arrays.append(pa.ExtensionArray.from_storage(PathArrowType(), array))
    else:
        new_arrays.append(pa.ExtensionArray.from_storage(PathArrowType(), new_array))
    table = table.add_column(table.num_columns,"shortest_paths", new_arrays)
    # output the files
    output_file = Path(config.output.file.name).expanduser()
    write_arrow(table, output_file)
    
"""
# batched addition of shortest_path.  Currently not implemented as ParquetFile.iter_batches()
# cannot handle nested data conversions for chunked array outputs.  Will be fixed in a future
# version of arrow.  See https://github.com/apache/arrow/issues/32723
     
     input_file = Path(config.input.file.name).expanduser()

    output = BatchWriter(config.output.file.name)
    # library_map = ArrowLibraryMap.from_parquet(input_file, num=config.input.num)
    with batch_reader(input_file) as reader:

        shortest_paths = []
        for table in reader:

            for i in range(len(table)):
                shortest_paths.append(jsonpickle.encode(get_shortest_paths(
                    table['mol'][i], config.conversion.max_path_length), keys=True))
            try:
                table = table.remove_column(table.column_names.index("shortest_paths"))
            except ValueError:
                pass
            new_array = pa.array(shortest_paths)
            if type(new_array) is pa.ChunkedArray:
                raise NotImplemented()
            else:
                new_array = pa.ExtensionArray.from_storage(PathArrowType(), new_array)
            table = table.add_column(table.num_columns,"shortest_paths", new_array)
            output.write_table(table)

    output.close()
    """

if __name__ == "__main__":
    path_generator_app()
