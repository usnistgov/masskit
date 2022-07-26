from massspec.peptide.encoding import calc_ions_mz, protonate_mass
import massspec.spectrum.spectrum as sp
import massspec.spectrum.join as mssj


def annotate_peptide_spectrum(spectrum, peptide=None, precursor_charge=None, ion_types=None, mod_names=None,
                              mod_positions=None):
    """
    annotate a spectrum with theoretically calculated ions

    """
    if peptide is None:
        peptide = spectrum.peptide
    if precursor_charge is None:
        precursor_charge = spectrum.charge
    if mod_names is None:
        mod_names = spectrum.mod_names
    if mod_positions is None:
        mod_positions = spectrum.mod_positions
    theo_spectrum = TheoreticalPeptideSpectrum(peptide, charge=precursor_charge, ion_types=ion_types,
                                               mod_names=mod_names, mod_positions=mod_positions)
    join = mssj.PairwiseJoin(spectrum, theo_spectrum).do_join()

    spectrum.add_join(join.results, theo_spectrum)


class TheoreticalSpectrum(sp.HiResSpectrum):
    """
    base class to contain a theoretical spectrum
    """

    def __init__(self, *args, **kwargs):
        """
        initialize predicted spectrum

        :param mz: array of mz values
        :param tolerance: mass tolerance in daltons
        :param count_spectra: count peaks instead of summing intensity
        :param take_max: when converting new_spectrum to evenly spaced bins, take max value per bin, otherwise sum
        :param analysis_annotations: turn on annotations used for peak analysis
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)


class TheoreticalPeptideSpectrum(TheoreticalSpectrum):
    """
    theoretical peptide spectrum
    """

    def __init__(self, peptide,
                 ion_types=None, mod_names=None, mod_positions=None,
                 analysis_annotations=False, num_isotopes=2, *args, **kwargs):
        """
        :param peptide: the peptide sequence
        :param ion_types: tuple or array of tuple of ion type and charge
        :param mod_names: any modifications
        :param mod_positions: the positions of the modifications
        :param analysis_annotations: add additional annotations useful for analyzing spectra
        :param num_isotopes: number of carbon-13 isotopes to calculate
        """
        super().__init__(*args, **kwargs)
        self.peptide = peptide
        self.peptide_len = len(peptide)
        self.mod_names = mod_names
        self.mod_positions = mod_positions

        
        if ion_types is None:
            ion_types = [("b", 1), ("b-H2O", 1), ("y", 1), ("y-H2O", 1), ("y-NH3", 1), ("a", 1), ("internalb", 1),
                         ("immonium", 1), ("parent", "z"), ("parent-H2O", "z"), ("parent-NH3", "z")]
            if self.charge > 1:
                ion_types.extend([("b", 2), ("y", 2), ("y", 3)])

            # check if phosphopeptide
            if mod_names is not None and 21 in mod_names:
                ion_types.extend([("b-H3PO4", 1), ("b-HPO3", 1), ("y-H3PO4", 1), ("y-HPO3", 1)])
        
        mz, intensity, annotations, precursor_mass = \
            calc_ions_mz(peptide, ion_types, mod_names=mod_names, mod_positions=mod_positions,
                         analysis_annotations=analysis_annotations, num_isotopes=num_isotopes)
        self.from_arrays(mz, intensity, product_mass_info=sp.MassInfo(0.0, "ppm", "monoisotopic"),
                         copy_arrays=False, annotations=annotations,
                         precursor_mz=protonate_mass(precursor_mass, self.charge), precursor_intensity=1.0,
                         precursor_mass_info=sp.MassInfo(0.0, "ppm", "monoisotopic"))
