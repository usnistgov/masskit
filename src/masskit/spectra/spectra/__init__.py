import copy
import logging
import re
from base64 import b64encode
from io import BytesIO, StringIO

import numpy as np
import pyarrow as pa

from .. import accumulator as _mkaccumulator
from ..data_specs import schemas as _mkschemas
from ..utils import fingerprints as _mkfingerprints
from ..utils import textalloc as _mktextalloc
from . import ipython as _mkipython
from . import spectrum_plotting as _mkspectrum_plotting
from . import ions as _mkions #import HiResIons, MassInfo, cosine_score_calc, nce2ev


class Spectrum:
    """
    Base class for spectrum with called ions.
    The props attribute is a dict that contains any structured data
    """

    def __init__(self, precursor_mass_info=None, product_mass_info=None, name=None, id=None,
                 ev=None, nce=None, charge=None, ion_class=_mkions.HiResIons, mz=None, intensity=None,
                 row=None, struct=None, precursor_mz=None, precursor_intensity=None, stddev=None,
                 annotations=None, tolerance=None, copy_arrays=False):
        """
        construct a spectrum.  Can initialize with arrays (mz, intensity) or arrow row object

        :param mz: mz array
        :param intensity: intensity array
        :param stddev: standard deviation of intensity
        :param row: dict containing parameters and precursor info or arrow row object
        :param struct: arrow struct containing parameters and precursor info
        :param precursor_mz: precursor_mz value, used preferentially to row
        :param precursor_intensity: precursor intensity, optional
        :param annotations: annotations on the ions
        :param precursor_mass_info: MassInfo mass measurement information for the precursor
        :param product_mass_info: MassInfo mass measurement information for the product
        :param tolerance: mass tolerance array
        :param copy_arrays: if the inputs are numpy arrays, make copies
        """

        self.joins = []  # join data structures
        self.joined_spectra = []  # corresponding spectra to joins
        self.props = {}
        self.prop_names = None
        self.precursor_class = ion_class
        self.product_class = ion_class
        if precursor_mass_info is None:
            self.precursor_mass_info = _mkions.MassInfo(20.0, "ppm", "monoisotopic", "", 1)
        else:
            self.precursor_mass_info = precursor_mass_info
        if product_mass_info is None:
            self.product_mass_info = _mkions.MassInfo(20.0, "ppm", "monoisotopic", "", 1)
        else:
            self.product_mass_info = product_mass_info
        self.name = name
        self.id = id
        self.charge = charge
        self.ev= ev
        self.nce=nce
        self.precursor = None
        self.products = None
        self.filtered = None  # filtered version of self.products

        if mz is not None and intensity is not None:
            self.from_arrays(
                mz,
                intensity,
                row=row,
                precursor_mz=precursor_mz,
                precursor_intensity=precursor_intensity,
                stddev=stddev,
                annotations=annotations,
                precursor_mass_info=precursor_mass_info,
                product_mass_info=product_mass_info,
                copy_arrays=copy_arrays,
                tolerance=tolerance,
                )
        elif row is not None:
            self.from_arrow(row, copy_arrays=copy_arrays)
        elif struct is not None:
            self.from_struct(struct, copy_arrays=copy_arrays)


    # def __getstate__(self):
    #     """
    #     omit members from pickle, jsonpickle, and deep copy
    #
    #     :return:
    #     """
    #     state = self.__dict__.copy()
    #     state.pop('_filtered', None)
    #     return state
    #
    # def __setstate__(self, state):
    #     self.__dict__.update(state)

    @property
    def filtered(self):
        """
        filtered version of product ions

        :return:
        """
        return self._filtered

    @filtered.setter
    def filtered(self, value):
        self._filtered = value
        return

    @property
    def props(self):
        """
        get property list

        :return: property list
        """
        return self._props

    @props.setter
    def props(self, value):
        self._props = value
        return

    def get_prop(self, name):
        """
        return a property from property list with None as default

        :param name: property name
        :return: value
        """
        return self.props.get(name)

    def get_props(self):
        """
        returns back all properties for this object

        :return: list of properties
        """
        return [p for p in dir(self.__class__) if isinstance(getattr(self.__class__, p), property)]

    def copy_props_from_dict(self, dict_in):
        """
        given a dictionary, e.g. a row, copy allowed properties in the spectrum props

        :param dict_in: the input dictionary.  Allowed properties are in self.prop_names
        """
        for prop in dict_in:
            if prop in self.get_props():
                self.props[prop] = copy.deepcopy(dict_in[prop])

    def add_join(self, join, joined_spectrum):
        """
        add a join and a corresponding joined spectrum

        :param join: the result of the join expressed as a arrow struct
        :param joined_spectrum: the spectrum object that is joined to this spectrum via the join
        """
        self.joins.append(join)
        self.joined_spectra.append(joined_spectrum)

    def from_arrow(self,
                   row,
                   copy_arrays=False):
        """
        Update or initialize from an arrow row object

        :param row: row object from which to extract the spectrum information
        :param copy_arrays: if the inputs are numpy arrays, make copies
        """

        # loop through the experimental fields and if there is data, save it to the spectrum
        for field in _mkschemas.property_fields:
            attribute = row.get(field.name)
            if attribute is not None:
                setattr(self, field.name, attribute())

        self.charge = row.charge() if row.get('charge') is not None else None

        stddev = row.stddev() if row.get('stddev') is not None else None
        annotations = row.annotations() if row.get('annotations') is not None else None
        precursor_intensity = row.precursor_intensity() if row.get('precursor_intensity') is not None else None

        self.precursor = self.precursor_class(
            mz=row.precursor_mz(),
            intensity=precursor_intensity,
            mass_info=_mkions.MassInfo(arrow_struct_accessor=row.precursor_massinfo),
        )

        tolerance = row.tolerance() if row.get('tolerance') is not None else None

        self.products = self.product_class(
            row.mz(),
            row.intensity(),
            stddev=stddev,
            mass_info=_mkions.MassInfo(arrow_struct_accessor=row.product_massinfo),
            annotations=annotations,
            copy_arrays=copy_arrays,
            tolerance=tolerance
        )
        return self

    def from_struct(self,
                   struct,
                   copy_arrays=False):
        """
        Update or initialize from an arrow struct object

        :param row: row object from which to extract the spectrum information
        :param copy_arrays: if the inputs are numpy arrays, make copies
        """

        # loop through the experimental fields and if there is data, save it to the spectrum
        for field in _mkschemas.property_fields:
            attribute = struct.get(field.name)
            if attribute is not None:
                if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
                    if attribute.values is not None:
                        # zero copy only is False to handle string lists
                        setattr(self, field.name, attribute.values.to_numpy(zero_copy_only=False))
                elif pa.types.is_struct(field.type):
                    if attribute.values is not None:
                        setattr(self, field.name, attribute.values)
                else:
                    setattr(self, field.name, attribute.as_py())

        self.charge = struct['charge'].as_py() if struct.get('charge') is not None else None

        def struct2numpy(struct, attribute):
            if struct.get(attribute) is not None:
                if struct[attribute].values is not None:
                    return struct[attribute].values.to_numpy()
            return None

        stddev = struct2numpy(struct, 'stddev')
        precursor_intensity = struct['precursor_intensity'].as_py() if struct.get('precursor_intensity') is not None else None

        self.precursor = self.precursor_class(
            mz=struct.get('precursor_mz'),
            intensity=precursor_intensity,
            mass_info=_mkions.MassInfo(arrow_struct_scalar=struct['precursor_massinfo']),
        )

        tolerance = struct2numpy(struct, 'tolerance')

        annotations = struct.get("annotations", None)
        if annotations is not None:
            # unwrap the struct array
            annotations = annotations.values

        self.products = self.product_class(
            struct2numpy(struct, 'mz'),
            struct2numpy(struct, 'intensity'),
            stddev=stddev,
            mass_info=_mkions.MassInfo(arrow_struct_scalar=struct['product_massinfo']),
            annotations=annotations,
            copy_arrays=copy_arrays,
            tolerance=tolerance
        )
        return self

    def from_arrays(
            self,
            mz,
            intensity,
            row=None,
            precursor_mz=None,
            precursor_intensity=None,
            stddev=None,
            annotations=None,
            precursor_mass_info=None,
            product_mass_info=None,
            copy_arrays=True,
            tolerance=None,
    ):
        """
        Update or initialize from a series of arrays and the information in rows.  precursor information
        is pulled from rows unless precursor_mz and/or procursor_intensity are provided.

        :param mz: mz array
        :param intensity: intensity array
        :param stddev: standard deviation of intensity
        :param row: dict containing parameters and precursor info
        :param precursor_mz: precursor_mz value, used preferentially to row
        :param precursor_intensity: precursor intensity, optional
        :param annotations: annotations on the ions
        :param precursor_mass_info: MassInfo mass measurement information for the precursor
        :param product_mass_info: MassInfo mass measurement information for the product
        :param tolerance: mass tolerance array
        :param copy_arrays: if the inputs are numpy arrays, make copies
        """

        if precursor_mass_info is None:
            precursor_mass_info = self.precursor_mass_info
        if product_mass_info is None:
            product_mass_info = self.product_mass_info

        # todo: we should turn these into properties for the spectrum that use what is in the product/precursor Ions
        self.precursor_mass_info = precursor_mass_info
        self.product_mass_info = product_mass_info

        if row:
            self.precursor = self.precursor_class(
                row.get("precursor_mz", None), mass_info=precursor_mass_info
            )
            self.name = row.get("name", None)
            self.id = row.get("id", None)
            self.retention_time = row.get("retention_time", None)
            # make a copy of the props
            self.props.update(row)
        if precursor_mz:
            self.precursor = self.precursor_class(
                precursor_mz, mass_info=precursor_mass_info
            )
        if precursor_intensity and self.precursor is not None:
            self.precursor.intensity = precursor_intensity
        # numpy array of peak intensity
        self.products = self.product_class(
            mz,
            intensity,
            stddev=stddev,
            mass_info=product_mass_info,
            annotations=annotations,
            copy_arrays=copy_arrays,
            tolerance=tolerance,
        )
        return self

    def get_string_prop(self, mol, prop_name):
        """
        read a molecular property, dealing with unicode decoding error (rdkit uses UTF-8)

        :param mol: the rdkit molecule
        :param prop_name: the name of the property
        :return: property value
        """
        prop = None
        if mol.HasProp(prop_name):
            try:
                prop = mol.GetProp(prop_name)
            except UnicodeDecodeError as err:
                logging.debug(
                    f"Invalid unicode character in property {prop_name} for spectrum {self.id} with error {err}"
                )
                prop = None
        return prop

    def get_float_prop(self, mol, prop_name):
        """
        read a molecular property, and parse it as a float, ignoring non number characters
        doesn't currently deal with exponentials

        :param mol: the rdkit molecule
        :param prop_name: the name of the property
        :return: property value
        """
        prop = None
        if mol.HasProp(prop_name):
            try:
                prop = mol.GetProp(prop_name)
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", prop)
                if matches:
                    prop = float(matches[0])
                else:
                    logging.debug(
                        f"No float in property {prop_name} for spectrum {self.id}"
                    )
                    prop = None
            except UnicodeDecodeError as err:
                logging.debug(
                    f"Invalid unicode character in property {prop_name} for spectrum {self.id} with error {err}"
                )
                prop = None
        return prop

    def get_int_prop(self, mol, prop_name):
        """
        read a molecular property, and parse it as a int, ignoring non number characters

        :param mol: the rdkit molecule
        :param prop_name: the name of the property
        :return: property value
        """
        prop = None
        if mol.HasProp(prop_name):
            try:
                prop = mol.GetProp(prop_name)
                matches = re.findall(r"[-+]?\d+", prop)
                if matches:
                    prop = int(matches[0])
                else:
                    logging.debug(
                        f"No int in property {prop_name} for spectrum {self.id}"
                    )
                    prop = None
            except UnicodeDecodeError as err:
                logging.debug(
                    f"Invalid unicode character in property {prop_name} for spectrum {self.id} with error {err}"
                )
                prop = None
        return prop

    def to_msp(self, annotate_peptide=False, ion_types=None):
        """
        convert a spectrum to an msp file entry, encoded as a string

        :param annotate_peptide: annotate as a peptide
        :param ion_types: ion types for annotation
        :return: string containing spectrum in msp format
        """
        #todo: check to see if annotation should be turned on
        if annotate_peptide:
            from masskit.spectra.theoretical_spectrum import \
                annotate_peptide_spectrum
            annotate_peptide_spectrum(self, ion_types=ion_types)

        ret_value = ""
        if hasattr(self, "name") and self.name is not None:
            ret_value += f"Name: {self.name}\n"
        else:
            ret_value += f"Name: {self.id}\n"
        if hasattr(self, "precursor") and self.precursor is not None and hasattr(self.precursor, "mz") and self.precursor.mz is not None:
            ret_value += f"PRECURSORMZ: {self.precursor.mz}\n"
        if hasattr(self, "formula") and self.formula is not None:
            ret_value += f"Formula: {self.formula}\n"
        if hasattr(self, "ev") and self.ev is not None:
            ret_value += f"eV: {self.ev}\n"
        if hasattr(self, "nce") and self.nce is not None:
            ret_value += f"NCE: {self.nce}\n"
        if hasattr(self, "protein_id") and self.protein_id is not None:
            ret_value += f"ProteinId: {','.join(self.protein_id)}\n"
        ret_value += f"DB#: {self.id}\n"
        # spectrum = self.copy(min_mz=1.0, min_intensity=min_intensity)
        num_peaks = len(self.products.mz)
        ret_value += f"Num Peaks: {num_peaks}\n"
        for i in range(num_peaks):
            ret_value += f"{self.products.mz[i]:.4f}\t{self.products.intensity[i]:.8g}"
            if self.joined_spectra and self.joins:
                ret_value += '\t"'
                # find join to annotation (for now, just one annotation to scan.  make join property handlers)
                # may be more than one annotation spectrum -- ignore this for now

                #   scan exp_peaks in join table, get matching theo_peaks
                annotation_strings = self.get_ion_annotation(i)
                if len(annotation_strings) > 0:
                    for k, annotation_string in enumerate(annotation_strings):
                        if k != 0:
                            ret_value += ","
                        ret_value += annotation_string
                else:
                    ret_value += '?'
                ret_value += '"\n'
            else:
                # if no annotation, print "?"
                ret_value += '\t"?"\n'
        ret_value += "\n"
        return ret_value

    def get_ion_annotation(self, i, show_ppm=True, tex_style=False, show_mz=True):
        """
        get the text annotations for a given ion

        :param i: index of the ion
        :param show_ppm: show a calcuated ppm
        :param tex_style: text annotations use tex for formatting
        :param show_mz: show mz values in the text
        """
        exp_peaks = self.joins[0].column('exp_peaks')
        theo_peaks = self.joins[0].column('theo_peaks')
        # from theo spectrum, get annotation struct.  see ion_annot_fields for fields
        annotations = self.joined_spectra[0].products.annotations
        ion_type = annotations.field(0)
        isotope = annotations.field(2)
        ion_subtype = annotations.field(3)
        position = annotations.field(4)
        end_position = annotations.field(5)
        product_charge = annotations.field(1)
        annotation_strings = []
        for j in range(len(exp_peaks)):
            if exp_peaks[j].is_valid and theo_peaks[j].is_valid and exp_peaks[j].as_py() == i:
                annotation_string = ""
                theo_peak = theo_peaks[j].as_py()
                current_ion_type = ion_type[theo_peak].as_py()
                current_ion_subtype = ion_subtype[theo_peak].as_py()

                if current_ion_type == 'internalb':
                    fragment = self.peptide[position[theo_peak].as_py(): end_position[theo_peak].as_py()]
                    annotation_string += f'Int/{fragment}'
                elif current_ion_type == 'immonium':
                    annotation_string += f'{current_ion_subtype}'
                elif current_ion_type == 'parent':
                    annotation_string += f'p'
                else:
                    if tex_style:
                        annotation_string += f'$\\mathregular{{{current_ion_type}_{{{position[theo_peak].as_py()}}}}}$'
                    else:
                        annotation_string += f'{current_ion_type}{position[theo_peak].as_py()}'

                if current_ion_subtype is not None and current_ion_type != 'immonium':
                    if tex_style:
                        # subscript atom counts
                        neutral_loss = re.sub(r"(\d+)", r"$\\mathregular{_{\1}}$", current_ion_subtype)
                        annotation_string += f'-{neutral_loss}'
                    else:
                        annotation_string += f'-{current_ion_subtype}'

                if isotope[theo_peak].as_py() == 1:
                    annotation_string += '+i'
                elif isotope[theo_peak].as_py() > 1:
                    annotation_string += f'+{isotope[theo_peak].as_py()}i'
                if product_charge[theo_peak].as_py() > 1 and current_ion_type != 'parent':
                    if tex_style:
                        annotation_string += f'$\\mathregular{{^{{{product_charge[theo_peak].as_py()}+}}}}$'
                    else:
                        annotation_string += f'^{product_charge[theo_peak].as_py()}'

                # calculate ppm
                if show_ppm:
                    ppm = (self.products.mz[i] - self.joined_spectra[0].products.mz[theo_peaks[j].as_py()]) / self.products.mz[i] * 1000000
                    annotation_string += f"/{ppm:.1f}ppm"

                if tex_style and show_mz:
                    annotation_string += f'\n{self.products.mz[i]:.2f}'
                annotation_strings.append(annotation_string)
        return annotation_strings

    def from_mol(
            self, mol, skip_expensive=False, id_field="NISTNO", id_field_type="int"
    ):
        """
        Initialize from rdkit mol

        :param mol: rdkit mol
        :param skip_expensive: skip the expensive calculations
        :param id_field: field to use for the mol id, such as NISTNO, ID or _NAME (the sdf title field)
        :param id_field_type: the id field type, such as int or str
        """

        if mol.HasProp("MW") or mol.HasProp("PRECURSOR M/Z"):
            if mol.HasProp("PRECURSOR M/Z"):
                precursor_mz = self.get_float_prop(mol, "PRECURSOR M/Z")
            else:
                precursor_mz = self.get_float_prop(mol, "MW")
            self.precursor = self.precursor_class(
                self.precursor_class.cast_mz(precursor_mz),
                mass_info=self.precursor_mass_info,
            )
        self.name = self.get_string_prop(mol, "NAME")
        self.casno = self.get_string_prop(mol, "CASNO")
        self.synonyms = self.get_string_prop(mol, "SYNONYMS")
        if self.synonyms is not None:
            self.synonyms = self.synonyms.splitlines()
        if type(id_field) is not int:
            if id_field_type == "int":
                self.id = self.get_int_prop(mol, id_field)
            else:
                self.id = self.get_string_prop(mol, id_field)

        if mol.HasProp("EXPERIMENTAL RI MEDIAN/DEVIATION/#DATA"):
            ri_string = mol.GetProp("EXPERIMENTAL RI MEDIAN/DEVIATION/#DATA")
            ris = ri_string.split()
            for ri_string in ris:
                ri = re.split("[=/]", ri_string)
                if len(ri) == 4:
                    if ri[0] == "SemiStdNP":
                        self.column = ri[0]
                        self.experimental_ri = float(ri[1])
                        self.experimental_ri_error = float(ri[2])
                        self.experimental_ri_data = int(ri[3])
                    elif ri[0] == "StdNP":
                        self.stdnp = float(ri[1])
                        self.stdnp_error = float(ri[2])
                        self.stdnp_data = int(ri[3])
                    elif ri[0] == "StdPolar":
                        self.stdpolar = float(ri[1])
                        self.stdpolar_error = float(ri[2])
                        self.stdpolar_data = int(ri[3])
        elif mol.HasProp("RIDATA_01"):
            self.column = self.get_string_prop(mol, "RIDATA_15")
            self.experimental_ri = self.get_float_prop(mol, "RIDATA_01")
            self.experimental_ri_error = 0.0
            self.experimental_ri_data = 1
        self.estimated_ri = self.get_float_prop(mol, "ESTIMATED KOVATS RI")
        self.inchi_key = self.get_string_prop(mol, "INCHIKEY")
        self.estimated_ri_error = self.get_float_prop(mol, "RI ESTIMATION ERROR")
        self.formula = self.get_string_prop(mol, "FORMULA")
        self.exact_mass = self.get_float_prop(mol, "EXACT MASS")
        self.ion_mode = self.get_string_prop(mol, "ION MODE")
        self.charge = self.get_int_prop(mol, "CHARGE")
        self.instrument = self.get_string_prop(mol, "INSTRUMENT")
        self.instrument_type = self.get_string_prop(mol, "INSTRUMENT TYPE")
        self.ionization = self.get_string_prop(mol, "IONIZATION")
        self.collision_gas = self.get_string_prop(mol, "COLLISION GAS")
        self.sample_inlet = self.get_string_prop(mol, "SAMPLE INLET")
        self.spectrum_type = self.get_string_prop(mol, "SPECTRUM TYPE")
        self.precursor_type = self.get_string_prop(mol, "PRECURSOR TYPE")
        notes = self.get_string_prop(mol, "NOTES")
        if notes is not None:
            match = re.search(r"Vial_ID=(\d+)", notes)
            if match:
                self.vial_id = int(match.group(1))
        ce = self.get_string_prop(mol, "COLLISION ENERGY")
        if ce is not None:
            match = re.search(r"NCE=(\d+)% (\d+)eV|NCE=(\d+)%|(\d+)|", ce)
            if match:
                if match.group(4) is not None:
                    self.ev = float(match.group(4))
                    self.collision_energy = float(self.ev)
                elif match.group(3) is not None:
                    self.nce = float(match.group(3))
                    if precursor_mz is not None and self.charge is not None:
                        self.collision_energy = _mkions.nce2ev(self.nce, precursor_mz, self.charge)
                elif match.group(1) is not None and match.group(2) is not None:
                    self.nce = float(match.group(1))
                    self.ev = float(match.group(2))
                    self.collision_energy = self.ev
        self.insource_voltage = self.get_int_prop(mol, "IN-SOURCE VOLTAGE")

        skip_props = set([
            "MASS SPECTRAL PEAKS",
            "NISTNO",
            "NAME",
            "MW",
            "EXPERIMENTAL RI MEDIAN/DEVIATION/#DATA",
            "ESTIMATED KOVATS RI",
            "INCHIKEY",
            "RI ESTIMATION ERROR",
            "FORMULA",
            "SYNONYMS",
            "RIDATA_01",
            "RIDATA_15",
            "PRECURSOR M/Z",
            "EXACT MASS",
            "ION MODE",
            "CHARGE",
            "INSTRUMENT",
            "INSTRUMENT TYPE",
            "IONIZATION",
            "COLLISION ENERGY",
            "COLLISION GAS",
            "SAMPLE INLET",
            "SPECTRUM TYPE",
            "PRECURSOR TYPE",
            "NOTES",
            "IN-SOURCE VOLTAGE",
            "ID",
        ])
        # populate spectrum props with mol props
        for k in mol.GetPropNames():
            # skip over props handled elsewhere
            if k in skip_props:
                continue
            self.props[k] = self.get_string_prop(mol, k)

        # get spectrum string and break into list of lines
        if mol.HasProp("MASS SPECTRAL PEAKS"):
            spectrum = mol.GetProp("MASS SPECTRAL PEAKS").splitlines()
            mz_in = []
            intensity_in = []
            annotations_in = []
            has_annotations = False
            for peak in spectrum:
                values = peak.replace("\n", "").split(
                    maxsplit=2
                )  # get rid of any newlines then strip
                intensity = self.product_class.cast_intensity(values[1])
                # there are intensity 0 ions in some spectra
                if intensity != 0:
                    mz_in.append(self.product_class.cast_mz(values[0]))
                    intensity_in.append(intensity)
                    annotations = []
                    if len(values) > 2 and not skip_expensive:
                        has_annotations = True
                        end = "".join(values[2:])
                        for match in re.finditer(
                                r"(\?|((\w+\d*)+((\+?-?)((\w|\d)+))?(=((\w+)(\+?-?))?(\w+\d*)+((\+?-?)((\w|\d)+))?)?("
                                r"\/(-?\d+.?\d+))(\w*)))*(;| ?(\d+)\/(\d+))",
                                end,
                        ):
                            annotation = {
                                "begin": match.group(10),
                                "change": match.group(11),
                                "loss": match.group(12),
                                "mass_diff": match.group(18),
                                "diff_units": match.group(19),
                                "fragment": match.group(3),
                            }
                            annotations.append(annotation)
                            # group 10 is the beginning molecule, e.g. "p" means precursor
                            # group 11 indicates addition or subtraction from the beginning molecule
                            # group 12 is the chemical formula of the rest of the beginning molecule
                            # group 18 is the mass difference, e.g. "-1.3"
                            # group 19 is the mass difference unit, e.g. "ppm"
                            # group 3 is the chemical formula of the peak
                    annotations_in.append(annotations)
            if has_annotations and not skip_expensive:
                logging.warning('spectrum annotations from molfile currently unimplemented')
                self.products = self.product_class(
                    mz_in,
                    intensity_in,
                    mass_info=self.product_mass_info,
                    # annotations=annotations_in,
                )
            else:
                self.products = self.product_class(
                    mz_in, intensity_in, mass_info=self.product_mass_info
                )
        return

    @staticmethod
    def weighted_intensity(intensity, mz):
        """
        Stein & Scott 94 intensity weight

        :param intensity: peak intensity
        :param mz: peak mz in daltons
        :return: weighted intensity
        """
        return intensity ** 0.6 * mz ** 3

    def single_match(self, spectrum, minimum_match=1, cosine_threshold=0.3, cosine_score_scale=1.0):
        """
        try to match two spectra and calculate the probability of a match

        :param spectrum: the spectrum to match against
        :param cosine_threshold: minimum score to return results
        :param minimum_match: minimum number of matching peaks
        :param cosine_score_scale: max value of cosine score
        :return: query id, hit id, cosine score, number of matching peaks
        """
        ion1, ion2, index1, index2 = self.products.clear_and_intersect(spectrum.products, None, None)

        # intersect the spectra
        matches = len(index1)
        if matches >= minimum_match:
            cosine_score = _mkions.cosine_score_calc(
                ion1.mz,
                ion1.intensity,
                ion2.mz,
                ion2.intensity,
                index1,
                index2,
                scale=cosine_score_scale,
            )
            if cosine_score >= cosine_threshold:
                return self.id, spectrum.id, cosine_score, matches
        return None, None, None, None

    def identity(self, spectrum, identity_name=False, identity_energy=False):
        """
        check to see if this spectrum and the passed in spectrum have the same chemical structure

        :param spectrum: comparison spectrum
        :param identity_name: require the name to match in addition to the inchi_key
        :param identity_energy: require the collision_energy to match in addition to the inchi_key
        :return: are they identical?
        """
        return_val = self.inchi_key == spectrum.inchi_key
        if self.inchi_key is None or spectrum.inchi_key is None:
            return_val = False
        if identity_name:
            return_val = return_val and (self.name == spectrum.name)
        if identity_energy:
            return_val = return_val and (
                    self.collision_energy == spectrum.collision_energy
            )
        return return_val

    def copy_annot(self, spectrum2):
        """
        copy annotations from spectrum2 to this spectrum using the matched ion indices

        :param spectrum2: the ions to compare against
        """
        index1, index2 = self.products.intersect(spectrum2.products)
        return self.products.copy_annot(spectrum2.products, index1, index2)

    def cosine_score(
            self,
            spectrum2,
            use_same_tolerance=False,
            mz_power=0.0,
            intensity_power=0.5,
            scale=999,
            skip_denom=False,
            tiebreaker=None
    ):
        """
        cosine score on product ions

        :param spectrum2: spectrum to compare against
        :param use_same_tolerance: evaluate cosine score by using the mass tolerance for this spectrum for spectrum2
        :param mz_power: what power to raise the mz value for each peak
        :param intensity_power: what power to raise the intensity for each peak
        :param scale: what value to scale the score by
        :param skip_denom: skip computing the denominator
        :param tiebreaker: how to deal with one to multiple matches to peaks in spectra1. mz is closest mz value, intensity is closest intensity, None is report multiple matches
        :return: cosine score
        """
        if use_same_tolerance:
            spectrum2 = spectrum2.change_mass_info(self.product_mass_info)
        return self.products.cosine_score(
            spectrum2.products,
            mz_power=mz_power,
            intensity_power=intensity_power,
            scale=scale,
            skip_denom=skip_denom,
            tiebreaker=tiebreaker
        )

    def intersect(self, spectrum2, tiebreaker=None):
        """
        intersect product ions with another spectrum's product ions

        :param spectrum2: comparison spectrum
        :param tiebreaker: how to deal with one to multiple matches to peaks in spectra1. mz is closest mz value, intensity is closest intensity, None is report multiple matches
        :return: matched ions indices in this spectrum, matched ion indices in comparison spectrum
        """
        return self.products.intersect(spectrum2.products, tiebreaker=tiebreaker)

    def copy(self, min_mz=-1, max_mz=0, min_intensity=-1, max_intensity=0):
        """
        create filtered version of self.  This is essentially a copy constructor

        :param min_mz: minimum mz value
        :param max_mz: maximum mz value.  0 = ignore
        :param min_intensity: minimum intensity value
        :param max_intensity: maximum intensity value.  0 = ignore
        :return: copy
        """
        return self.filter(
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
            inplace=False,
        )

    def filter(
            self, min_mz=-1, max_mz=0, min_intensity=-1, max_intensity=0, inplace=False
    ):
        """
        filter a spectrum by mz and/or intensity.

        :param min_mz: minimum mz value
        :param max_mz: maximum mz value.  0 = ignore
        :param min_intensity: minimum intensity value
        :param max_intensity: maximum intensity value.  0 = ignore
        :param inplace: do operation on current spectrum, otherwise create copy
        :return: filtered copy if not inplace, otherwise current ions
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.filter(
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            max_intensity=max_intensity,
            inplace=True,
        )
        # zero out annotations as they are no longer valid
        return_spectrum.joins = []
        return_spectrum.joined_spectra = []
        return return_spectrum

    def parent_filter(self, h2o=True, inplace=False):
        """
        filter parent ions, including water losses.

        :param h2o: filter out water losses
        :param inplace: do operation on current ions, otherwise create copy
        :return: filtered copy if not inplace, otherwise current ions
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        if self.precursor is not None:
            return_spectrum.products.parent_filter(h2o=h2o, precursor_mz=self.precursor.mz, charge=self.charge,
                                                   inplace=True)
        # zero out annotations as they are no longer valid
        return_spectrum.joins = []
        return_spectrum.joined_spectra = []
        return return_spectrum

    def windowed_filter(self, mz_window=14, num_ions=5, inplace=False):
        """
        filter ions by examining peaks in order of intensity and filtering peaks within a window

        :param mz_window: half size of mz_window for filtering
        :param num_ions: number of ions allowed in full mz_window
        :param inplace: do operation on current ions, otherwise create copy
        :return: filtered copy if not inplace, otherwise current ions
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.windowed_filter(mz_window=mz_window, num_ions=num_ions, inplace=True)

        # zero out annotations as they are no longer valid
        return_spectrum.joins = []
        return_spectrum.joined_spectra = []
        return return_spectrum

    def norm(self, max_intensity_in=999, keep_type=True, inplace=False, ord=None):
        """
        norm the product intensities

        :param max_intensity_in: the intensity of the most intense peak
        :param keep_type: keep the type of the intensity array
        :param inplace: do operation on current spectrum, otherwise create copy
        :param ord: if set, normalize using norm order as in np.linalg.norm. 2 = l2
        :returns: normed copy if not inplace, otherwise current ions
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.norm(
            max_intensity_in=max_intensity_in, keep_type=keep_type, inplace=True, ord=ord
        )
        return return_spectrum

    def merge(self, merge_spectrum, inplace=False):
        """
        merge the product ions of another spectrum into this one.

        :param merge_spectrum: the spectrum whose product ions will be merged in
        :param inplace: do operation on current spectrum
        :returns: merged copy if not inplace, otherwise current spectrum
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.merge(merge_spectrum.products, inplace=True)

        # zero out annotations as they are no longer valid
        return_spectrum.joins = []
        return_spectrum.joined_spectra = []
        return return_spectrum

    def shift_mz(self, shift, inplace=False):
        """
        shift the mz values of all product ions by the value of shift.  Negative ions are masked out

        :param shift: value to shift mz
        :param inplace: do operation on current ions
        :returns: masked copy if not inplace, otherwise current ions
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.shift_mz(shift=shift, inplace=True)
        return return_spectrum

    def mask(self, indices, inplace=False):
        """
        mask out product ions that are pointed to by the indices

        :param indices: indices of ions to screen out or numpy boolean mask
        :param inplace: do operation on current ions
        :returns: masked copy if not inplace, otherwise current spectrum
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.mask(indices=indices, inplace=True)

        # zero out annotations as they are no longer valid
        return_spectrum.joins = []
        return_spectrum.joined_spectra = []
        return return_spectrum

    def change_mass_info(self, mass_info, inplace=False, take_max=True):
        """
        given a new mass info for product ions, recalculate tolerance bins

        :param mass_info: the MassInfo structure to change to
        :param inplace: if true, change in place, otherwise return copy
        :param take_max: for each bin take the maximum intensity ion, otherwise sum all ions mapping to the bin
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)
        return_spectrum.products.change_mass_info(
            mass_info, inplace=True, take_max=take_max
        )
        return return_spectrum

    def total_intensity(self):
        """
        total intensity of product ions

        :return: total intensity
        """
        return self.products.total_intensity()

    def num_ions(self):
        """
        number of product ions

        :return: number of ions
        """
        return self.products.num_ions()

    def plot(
            self,
            axes=None,
            mirror_spectrum=None,
            mirror=True,
            title=None,
            xlabel="m/z",
            ylabel="Intensity",
            title_size=None,
            label_size=None,
            max_mz=None,
            min_mz=0,
            color=(0, 0, 1, 1),
            mirror_color=(1, 0, 0, 1),
            stddev_color=(0.3, 0.3, 0.3, 0.5),
            left_label_color=(1, 0, 0, 1),
            normalize=None,
            plot_stddev=False,
            vertical_cutoff=0.0,
            vertical_multiplier=1.1,
            right_label=None,
            left_label=None,
            right_label_size=None,
            left_label_size=None,
            no_xticks=False,
            no_yticks=False,
            linewidth=None,
            annotate=False,
        ):
        """
        make a spectrum plot using matplotlib.  if mirror_spectrum is specified, will do a mirror plot

        :param axes: matplotlib axis
        :param mirror_spectrum: spectrum to mirror plot (optional)
        :param mirror: if true, mirror the plot if there are two spectra.  Otherwise plot the two spectra together
        :param title: title of plot
        :param xlabel: xlabel of plot
        :param ylabel: ylabel of plot
        :param title_size: size of title font
        :param label_size: size of x and y axis label fonts
        :param max_mz: maximum mz to plot
        :param min_mz: minimum mz to plot
        :param color: color of spectrum specified as RBGA tuple
        :param mirror_color: color of mirrored spectrum specified as RGBA tuple
        :param stddev_color: color of error bars
        :param left_label_color: color of the left top label
        :param normalize: if specified, norm the spectra to this value.
        :param plot_stddev: if true, plot the standard deviation
        :param vertical_cutoff: if the intensity/max_intensity is below this value, don't plot the vertical line
        :param vertical_multiplier: multiply times y max values to create white space
        :param right_label: label for the top right of the fiture
        :param left_label: label for the top left of the figure
        :param right_label_size: size of label for the top right of the fiture
        :param left_label_size: size of label for the top left of the figure
        :param no_xticks: turn off x ticks and labels
        :param no_yticks: turn off y ticks and lables
        :param linewidth: width of plotted lines
        :param annotate: if peptide spectra, annotate ions
        :return: peak_collection, mirror_peak_collection sets of peaks for picking
        """
        #TODO: move this code into plotting.py so that it can use matplotlib, e.g use the default axes.
        if mirror_spectrum:
            mirror_ions = mirror_spectrum.products
            mirror_intensity = mirror_ions.intensity
            mirror_mz = mirror_ions.mz
            if plot_stddev:
                mirror_stddev = mirror_ions.stddev
            else:
                mirror_stddev = None
        else:
            mirror_ions = None
            mirror_intensity = None
            mirror_mz = None
            mirror_stddev = None

        if plot_stddev:
            stddev = self.products.stddev
        else:
            stddev = None

        line_collections = _mkspectrum_plotting.spectrum_plot(
            axes,
            self.products.mz,
            self.products.intensity,
            stddev,
            mirror_mz=mirror_mz,
            mirror_intensity=mirror_intensity,
            mirror_stddev=mirror_stddev,
            mirror=mirror,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            title_size=title_size,
            label_size=label_size,
            max_mz=max_mz,
            min_mz=min_mz,
            color=color,
            mirror_color=mirror_color,
            stddev_color=stddev_color,
            left_label_color=left_label_color,
            normalize=normalize,
            vertical_cutoff=vertical_cutoff,
            vertical_multiplier=vertical_multiplier,
            right_label=right_label,
            left_label=left_label,
            right_label_size=right_label_size,
            left_label_size=left_label_size,
            no_xticks=no_xticks,
            no_yticks=no_yticks,
            linewidth=linewidth
        )

        if annotate:
            annots = []
            xx = []
            yy = []
            for j in range(len(self.products)):
                annot = self.get_ion_annotation(j, show_ppm=False, tex_style=True, show_mz=False)
                if len(annot) > 0:
                    annots.append(annot[0])
                    xx.append(self.products.mz[j])
                    yy.append(self.products.intensity[j])

            _mktextalloc.allocate_text(axes,xx,yy,
                            annots,
                            x_lines=[np.array([self.products.mz[i],self.products.mz[i]]) for i in range(len(self.products))],
                            y_lines=[np.array([0,self.products.intensity[i]]) for i in range(len(self.products))],
                            textsize=7,
                            margin=0.0,
                            min_distance=0.005,
                            max_distance=0.05,
                            linewidth=0.7,
                            nbr_candidates=200,
                            textcolor="black",
                            draw_all=False,
                            ylims=(axes.get_ylim()[1]/20, axes.get_ylim()[1]))

        return line_collections


    def __repr__(self):
        """
        text representation of spectrum

        :return: text
        """
        if self.precursor is not None:
            return f"<spectrum {self.id}; {self.precursor.mz}Da precursor; {self.num_ions()} ions>"
        elif self.products is not None:
            return f"<spectrum {self.id}; {self.num_ions()} ions>"
        else:
            return f"<spectrum {self.id}>"

    def draw_spectrum(self, fig_format, output):
        return _mkspectrum_plotting.draw_spectrum(self, fig_format, output)

    def _repr_png_(self):
        """
        png representation of spectrum

        :return: png
        """
        return self.draw_spectrum("png", BytesIO())

    def _repr_svg_(self):
        """
        png representation of spectrum

        :return: svg
        """
        return self.draw_spectrum("svg", StringIO())

    def __str__(self):
        if _mkipython.is_notebook():
            val = b64encode(self._repr_png_()).decode("ascii")
            return \
                f'<img data-content="masskit/spectrum" src="data:image/png;base64,{val}" alt="spectrum {self.name}"/>'
        else:
            return self.__repr__()

    def create_fingerprint(self, max_mz=2000):
        """
        create a fingerprint that corresponds to the spectra.  Each bit position
        corresponds to an integer mz value and is set if the intensity is above min_intensity

        :param max_mz: the length of the fingerprint (also corresponds to maximum mz value)
        :return: SpectrumTanimotoFingerPrint
        """
        fp = _mkfingerprints.SpectrumTanimotoFingerPrint(dimension=max_mz)
        fp.object2fingerprint(self)
        return fp

    def finalize(self):
        """
        function used to clean up spectrum after creation
        """
        pass

    def evenly_space(self, tolerance=None, take_max=True, max_mz=None, include_zeros=False, inplace=False,
                    take_sqrt=False):
        """
        convert product ions to product ions with evenly spaced m/z bins.  The m/z bins are centered on
        multiples of tolerance * 2.  Multiple ions that map to the same bin are either summed or the max taken of the
        ion intensities.

        :param tolerance: the mass tolerance of the evenly spaced m/z bins (bin width is twice this value) in daltons
        :param take_max: for each bin take the maximum intensity ion, otherwise sum all ions mapping to the bin
        :param max_mz: maximum mz value, 2000 by default
        :param include_zeros: fill out array including bins with zero intensity
        :param inplace: do operation on current spectrum, otherwise create copy
        :param take_sqrt: take the sqrt of the intensities
        :returns: normed copy if not inplace, otherwise current ions
        """
        if inplace:
            return_spectrum = self
        else:
            return_spectrum = copy.deepcopy(self)

        return_spectrum.products.evenly_space(tolerance=tolerance, take_max=take_max, max_mz=max_mz,
                                                include_zeros=include_zeros, take_sqrt=take_sqrt)
        return return_spectrum
    
    # Add properties from the schema to Spectrum
_mkschemas.populate_properties(Spectrum)


class AccumulatorSpectrum(Spectrum, _mkaccumulator.Accumulator):
    """
    used to contain a spectrum that accumulates the sum of many spectra
    includes calculation of standard deviation
    """
    prop_names = None

    def __init__(self, mz=None, tolerance=None, precursor_mz=None, count_spectra=False, take_max=True, *args, **kwargs):
        """
        initialize predicted spectrum

        :param mz: array of mz values
        :param tolerance: mass tolerance in daltons
        :param precursor_mz: m/z of precursor
        :param count_spectra: count peaks instead of summing intensity
        :param take_max: when converting new_spectrum to evenly spaced bins, take max value per bin, otherwise sum
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        # keep count of the number of spectra being averaged into each bin
        self.count = np.zeros_like(mz)
        self.from_arrays(mz, np.zeros_like(mz), stddev=np.zeros_like(mz),
                         product_mass_info=_mkions.MassInfo(tolerance, "daltons", "monoisotopic", evenly_spaced=True),
                         precursor_mz=precursor_mz, precursor_intensity=999.0,
                         precursor_mass_info=_mkions.MassInfo(0.0, "ppm", "monoisotopic"))
        self.count_spectra = count_spectra
        self.take_max = take_max

    def add(self, new_item):
        """
        add a spectrum to the average.  Keeps running total of average and std deviation using
        Welford's algorithm.  This API assumes that the spectrum being added is also evenly spaced
        with the same mz values.  However, the new spectrum doesn't have to have the same max_mz as
        the summation spectrum

        :param new_item: new spectrum to be added
        """
        # convert to same scale
        intensity = np.zeros((1, len(self.products.intensity)))
        new_item.products.ions2array(
            intensity,
            0,
            bin_size=self.products.mass_info.tolerance * 2,
            down_shift=self.products.mass_info.tolerance,
            intensity_norm=1.0,
            channel_first=True,
            take_max=self.take_max
        )
        intensity = np.squeeze(intensity)
        if self.count_spectra:
            intensity[intensity > 0.0] = 1.0

        # assume all spectra start at 0, but may have different max_mz.  len_addition is the size of the smallest
        # mz array
        if len(intensity) >= len(self.products.mz):
            len_addition = len(self.products.mz)
        else:
            len_addition = len(intensity)

        delta = intensity[0:len_addition] - self.products.intensity[0:len_addition]
        # increment the count, dealing with case where new spectrum is longer than the summation spectrum
        self.count[0:len_addition] += 1
        self.products.intensity[0:len_addition] += delta / self.count[0:len_addition]
        delta2 = intensity[0:len_addition] - self.products.intensity[0:len_addition]
        self.products.stddev[0:len_addition] += delta * delta2

    def finalize(self):
        """
        finalize the std deviation after all the the spectra have been added
        """
        self.products.stddev = np.sqrt(self.products.stddev / self.count)

        # delete extra attributes and change class to base class
        del self.count
        del self.count_spectra
        del self.take_max
        self.__class__ = Spectrum

_mkschemas.populate_properties(AccumulatorSpectrum, fields=_mkschemas.spectrum_accumulator_fields)


class HiResSpectrum(Spectrum):
    def __init__(self, precursor_mass_info=None, product_mass_info=None, name=None, id=None, ev=None, nce=None, charge=None, ion_class=_mkions.HiResIons, mz=None, intensity=None, row=None, precursor_mz=None, precursor_intensity=None, stddev=None, annotations=None, tolerance=None, copy_arrays=False):
        super().__init__(precursor_mass_info=precursor_mass_info,
                         product_mass_info=product_mass_info,
                         name=name,
                         id=id,
                         ev=ev,
                         nce=nce,
                         charge=charge,
                         ion_class=ion_class,
                         mz=mz,
                         intensity=intensity,
                         row=row,
                         precursor_mz=precursor_mz,
                         precursor_intensity=precursor_intensity,
                         stddev=stddev,
                         annotations=annotations,
                         tolerance=tolerance,
                         copy_arrays=copy_arrays,
                         )