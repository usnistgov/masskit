import csv
import numpy as np
from abc import ABC, abstractmethod
from masskit.spectrum.spectrum import Spectrum
from masskit.data_specs.schemas import min_spectrum_fields, property_fields
import masskit.utils.files as msuf
from masskit.utils.general import open_if_filename
from masskit.utils.tables import row_view, arrow_to_pandas
import jsonpickle
from rdkit import Chem

"""
TableMap classes

Mimics pytorch dataset and used to iterate over a dataset and retrieve rows of the dataset
as a dictionary.
"""

class TableMap(ABC):
    """
    collections.abc.Sequence wrapper for a library.  Allows use of different stores, e.g. arrow or pandas
    """
    def __init__(self, column_name=None, *args, **kwargs):
        """
        :param column_name: the column name for the spectrum. "spectrum" is default
        """
        super().__init__(*args, **kwargs)
        if column_name is None:
            self.column_name = "spectrum"
        else:
            self.column_name = column_name
        # fields to report
        self.field_list = property_fields

    def __getitem__(self, key):
        """
        get spectrum from the library by row number

        :param key: row number
        :return: spectrum at row number
        """
        return self.getitem_by_row(key)

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def create_dict(self, idx):
        """
        create dict for row

        :param idx: row number
        """
        raise NotImplementedError

    @abstractmethod
    def get_ids(self):
        """
        get the ids of all records, in row order

        :return: array of ids
        """
        raise NotImplementedError

    @abstractmethod
    def getrow_by_id(self, key):
        """
        given an id, return the corresponding row number in the table

        :param key: the id
        :return: the row number of the id in the table
        """
        return NotImplementedError

    @abstractmethod
    def getitem_by_id(self, key):
        """
        get an item from the library by id

        :param key: id
        :return: the row dict
        """
        raise NotImplementedError

    @abstractmethod
    def getitem_by_row(self, key):
        """
        get an item from the library by id
        :param key: row number
        :return: the row dict
        """
        raise NotImplementedError
    
    def to_msp(self, file, annotate=False, ion_types=None):
        """
        write out spectra in msp format

        :param file: file or filename to write to
        :param annotate: annotate the spectra
        :param ion_types: ion types for annotation
        """
        spectra = [self[i][self.column_name] for i in range(len(self))]
        msuf.spectra_to_msp(file, spectra, annotate=annotate, ion_types=ion_types)

"""
code to convert encoded or multicolumn data structures into python objects.

It would be useful to (a) include the names of these fields in the TableMap field_names,
(b) have converters to str format for output to csv files, (c) possibly have this work from tables.py
"""

def make_spectrum(row):
    return Spectrum(row=row)

def make_mol(row):
    attribute = row.get('mol')
    if attribute is not None:
        value = attribute()
        if value is not None:
            return Chem.rdMolInterchange.JSONToMols(value)[0]
    else:
        return None

def make_shortest_paths(row):
    attribute = row.get('shortest_paths')
    if attribute is not None:
        value = attribute()
        if value is not None:
            return jsonpickle.decode(value, keys=True)
    else:
        return None

converter_list = {
    'spectrum': make_spectrum,
    'mol': make_mol,
    'shortest_paths': make_shortest_paths,
}

class ArrowLibraryMap(TableMap):
    """
    wrapper for an arrow library

    """

    def __init__(self, table_in, column_name=None, num=0, conversions=None, *args, **kwargs):
        """
        :param table_in: parquet table
        :param column_name: name of the spectrum column
        :param num: number of rows to use
        :param conversions: conversions when converting to pandas dataframe. None=['spectrum']
        """
        super().__init__(column_name=column_name, *args, **kwargs)
        self.table = table_in
        if num:
            self.table = self.table.slice(0, num)
        self.row = row_view(self.table)
        self.length = len(self.table['id'])
        self.ids = self.table['id'].combine_chunks().to_numpy()
        self.sort_indices = np.argsort(self.ids)
        self.sorted_ids = self.ids[self.sort_indices]
        if conversions == None:
            self.conversions = ['spectrum']
        else:
            self.conversions = conversions

    def __len__(self):
        return self.length

    def get_ids(self):
        return self.ids

    def getrow_by_id(self, key):
        pos = np.searchsorted(self.sorted_ids, key)
        if pos > self.length or key != self.sorted_ids[pos]:
            raise IndexError(f'unable to find key {key}')
        return self.sort_indices[pos]

    def create_dict(self, idx):
        self.row.idx = idx
        return_val = {}
        # put interesting fields in the dictionary
        for field in self.field_list:
            if field.name in converter_list:
                converted_val = converter_list[field.name](self.row)
                if converted_val is not None:
                    return_val[field.name] = converted_val
            else:
                attribute = self.row.get(field.name)
                if attribute is not None:
                    return_val[field.name] = attribute()

        # create the spectrum
        if self.column_name is not None:
            return_val[self.column_name] = Spectrum(row=self.row)
        return return_val

    def getitem_by_id(self, key):
        pos = self.getrow_by_id(key)
        return self.create_dict(pos)

    def getitem_by_row(self, key):
        # need to expand to deal with slice object
        if isinstance(key, slice):
            print(key.start, key.stop, key.step)
            raise NotImplementedError
        else:
            assert (0 <= key < len(self))
            return self.create_dict(key)

    def to_arrow(self):
        return self.table
    
    def to_pandas(self):
        return arrow_to_pandas(self.table, conversions=self.conversions, field_list=self.field_list)

    def to_parquet(self, file):
        """
        save spectra to parquet file

        :param file: filename or stream
        """
        msuf.write_parquet(file, self.table)
        
    def to_mzxml(self, file, use_id_as_scan=True):
        """
        save spectra to mzxml format file

        :param file: filename or stream
        :param use_id_as_scan: use spectrum.id instead of spectrum.scan
        """
        spectra = [self[i][self.column_name] for i in range(len(self))]
        msuf.spectra_to_mzxml(file, spectra, use_id_as_scan=use_id_as_scan)
            
    def to_mgf(self, file):
        """
        save spectra to mgf file

        :param file: filename or file pointer
        """
        spectra = [self[i][self.column_name] for i in range(len(self))]
        msuf.spectra_to_mgf(file, spectra)

    def to_csv(self, file, columns=None):
        """
        Write to csv file, skipping any spectrum column and writing mol columns as
        canonical SMILES

        :param file: filename or file pointer.  newline should be set to ''
        :param columns: list of columns to write out to csv file.  If none, all columns
        """
        if columns is None:
            columns = self.field_list

        fp = open_if_filename(file, 'w', newline='')
        csv_writer = csv.DictWriter(fp, fieldnames=columns, extrasaction='ignore')
        csv_writer.writeheader()

        for i in range(len(self)):
            row = self.getitem_by_row(i)
            if 'mol' in row:
                row['mol'] = Chem.rdmolfiles.MolToSmiles(row['mol'])
            csv_writer.writerow(row)

    @staticmethod
    def from_parquet(file, columns=None, num=None, combine_chunks=False, filters=None):
        """
        create an ArrowLibraryMap from a parquet file

        :param file: filename or stream
        :param columns: list of columns to read.  None=all, []=minimum set
        :param num: number of rows
        :param combine_chunks: dechunkify the arrow table to allow zero copy
        :param filters: parquet predicate as a list of tuples
        """
        if columns is not None:
            columns = list(set(columns + min_spectrum_fields))
        input_table = msuf.read_parquet(file, columns=columns, num=num, filters=filters)
        if len(input_table) == 0:
            raise IOError(f'Parquet file {file} read in with zero rows when using filters {filters}')
        if combine_chunks:
            input_table = input_table.combine_chunks()
        return ArrowLibraryMap(input_table)

    @staticmethod
    def from_msp(file, num=None, id_field=0, comment_fields=None, min_intensity=0.0, max_mz=2000):
        """
        read in an msp file and create an ArrowLibraryMap

        :param file: filename or stream
        :param num: number of rows.  None means all
        :param id_field: start value of the id field
        :param comment_fields: a Dict of regexes used to extract fields from the Comment field.  Form of the Dict is { comment_field_name: (regex, type, field_name)}.  For example {'Filter':(r'@hcd(\d+\.?\d* )', float, 'nce')}
        :param min_intensity: the minimum intensity to set the fingerprint bit
        :param max_mz: the length of the fingerprint (also corresponds to maximum mz value)
        :return: ArrowLibraryMap
        """
        return ArrowLibraryMap(msuf.load_msp2array(file, num=num, id_field=id_field, comment_fields=comment_fields,
                                              min_intensity=min_intensity, max_mz=max_mz))

    @staticmethod
    def from_mgf(file, num=None, title_fields=None, min_intensity=0.0, max_mz=2000):
        """
        read in an mgf file and create an ArrowLibraryMap

        :param file: filename or stream
        :param num: number of rows.  None means all
        :param title_fields: dict containing column names with corresponding regex to extract field values from the TITLE
        :param min_intensity: the minimum intensity to set the fingerprint bit
        :param max_mz: the length of the fingerprint (also corresponds to maximum mz value)
        :return: ArrowLibraryMap
        """
        return ArrowLibraryMap(msuf.load_mgf2array(file, num=num, title_fields=title_fields,
                                                   min_intensity=min_intensity, max_mz=max_mz))


    @staticmethod
    def from_sdf(file,
                 num=None,
                 skip_expensive=True,
                 max_size=0,
                 source=None,
                 id_field=None,
                 min_intensity=0.0,
                 max_mz=2000,
                 set_probabilities=(0.01, 0.97, 0.01, 0.01),
        ):
        """
        read in an sdf file and create an ArrowLibraryMap

        :param file: filename or stream
        :param num: number of rows.  None means all
        :param skip_expensive: don't compute fields that are computationally expensive
        :param max_size: the maximum bounding box size (used to filter out large molecules. 0=no bound)
        :param source: where did the sdf come from?  pubchem, nist, ?
        :param id_field: field to use for the mol id, such as NISTNO, ID or _NAME (the sdf title field). if an integer,
          use the integer as the starting value for an assigned id
        :param min_intensity: the minimum intensity to set the fingerprint bit
        :param max_mz: the length of the fingerprint (also corresponds to maximum mz value)
        :param set_probabilities: how to divide into dev, train, valid, test
        :return: ArrowLibraryMap
        """
        return ArrowLibraryMap(msuf.load_sdf2array(file, num=num, skip_expensive=skip_expensive, max_size=max_size,
                                              source=source, id_field=id_field, min_intensity=min_intensity,
                                              max_mz=max_mz, set_probabilities=set_probabilities))


class PandasLibraryMap(TableMap):
    """
    wrapper for a pandas spectral library

    """

    def __init__(self, df, column_name=None, *args, **kwargs):
        """
        :param df: pandas dataframe
        :param column_name: name of the spectrum column
        """
        super().__init__(column_name=column_name, *args, **kwargs)
        self.df = df
        self.ids = self.df.index.values
        self.length = len(self.ids)
        self.sort_indices = np.argsort(self.ids)
        self.sorted_ids = self.ids[self.sort_indices]

    def __len__(self):
        return self.length

    def create_dict(self, idx):
        return_val = self.df.iloc[[idx]].to_dict(orient='records')[0]
        return_val[self.df.index.name] = self.df.index.values[idx]
        return return_val

    def get_ids(self):
        return self.ids

    def getrow_by_id(self, key):
        pos = np.searchsorted(self.sorted_ids, key)
        if pos > self.length or key != self.sorted_ids[pos]:
            raise IndexError(f'unable to find key {key}')
        return self.sort_indices[pos]

    def getitem_by_id(self, key):
        pos = self.getrow_by_id(key)
        return self.create_dict(pos)

    def getitem_by_row(self, key):
        return self.create_dict(key)

class ListLibraryMap(TableMap):
    """
    wrapper for a spectral library using python lists

    """

    def __init__(self, list_in,  *args, **kwargs):
        """
        :param list_in: list of spectra
        """
        super().__init__(*args, **kwargs)
        self.list_in = list_in
        self.length = len(self.list_in)

    def __len__(self):
        return self.length

    def create_dict(self, idx):
        spectrum = self.list_in[idx]
        return_val = {}
        # put interesting fields in the dictionary
        for field in self.field_list:
            try:
                attribute = getattr(spectrum, field.name)
                if attribute is not None:
                    return_val[field.name] = attribute
            except AttributeError:
                pass

        return_val[self.column_name] = spectrum
        return return_val

    def get_ids(self):
        return [x for x in range(self.length)]

    def getrow_by_id(self, key):
        return key

    def getitem_by_id(self, key):
        return self.create_dict(key)

    def getitem_by_row(self, key):
        return self.create_dict(key)
