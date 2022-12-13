import re
from abc import ABC, abstractmethod
import string
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pandas as pd
from masskit.data_specs.schemas import ion_annot_fields
import masskit.spectrum.spectrum as mss
import masskit.utils.index as msui


class Join(ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = None

    @abstractmethod
    def do_join(self, tiebreaker="mz"):
        """
        do the join

        :param tiebreaker: how to deal with one to multiple matches. mz is closest mz value, intensity is closest intensity, None is report multiple matches
        :return: self
        """
        raise NotImplementedError

    @staticmethod
    def join_2_spectra(spectra1, spectra2, tiebreaker="mz"):
        """
        left join of two spectra.  iterate through all peaks for spectra1 and return joined spectra2 peaks.
        results also include unmatched spectra1 (left) peaks

        :param spectra1: first spectra. all peaks included in result
        :param spectra2: second spectra
        :param tiebreaker: how to deal with one to multiple matches to peaks in spectra1. mz is closest mz value, intensity is closest intensity, None is report multiple matches
        :return: list of peak ids from spectrum1, list of peak ids from spectrum2
        """
        index1, index2 = spectra1.intersect(spectra2)
        return mss.dedup_matches(spectra1.products, spectra2.products, index1, index2, tiebreaker=tiebreaker, skip_nomatch=False)

    @staticmethod
    def join_3_spectra(experimental_spectrum, predicted_spectrum, theoretical_spectrum, tiebreaker="mz"):
        """
        Join the peaks in a single experimental spectrum to a predicted spectrum and a theoretical spectrum.
        The join lists returned include all experimental and predicted peaks, but only the theoretical
        peaks that match the experimental spectra (and not necessarily the predicted spectrum).
        Note that it is possible to get a join where the theoretical peak matches the experimental peak but
        not the predicted peak.

        :param experimental_spectrum: the experimental spectrum
        :param predicted_spectrum: the predicted spectrum
        :param theoretical_spectrum: the annotated theoretical spectrum
        :param tiebreaker: how to deal with one to multiple matches. mz is closest mz value, intensity is closest intensity, None is report multiple matches
        :return: 3 lists with the peak ids. first is experimental peaks matching the theoretical peaks. Second are
        the predicted peaks that match the experimental peaks. Third are the theoretical peaks that match the
        experimental peaks.  A value of None indicates no join.
        """

        # naming convention of the peak id lists: first name after "join_" denotes the spectrum for the peak ids
        join_exp_x_theo_exp, join_exp_x_theo_theo = Join.join_2_spectra(experimental_spectrum, theoretical_spectrum, tiebreaker=tiebreaker)
        join_pred_x_exp_pred, join_pred_x_exp_exp = Join.join_2_spectra(predicted_spectrum, experimental_spectrum, tiebreaker=tiebreaker)

        # holds pred peak that match exp peaks that are joined to theo peaks
        join_pred_x_exp_theo = [None] * len(join_exp_x_theo_exp)

        # go through the exp peaks in left join_exp_x_theo
        for i in range(len(join_exp_x_theo_exp)):
            try:
                # look to see if there is a matching experimental to theo join
                pos = join_pred_x_exp_exp.index(join_exp_x_theo_exp[i])
                # if there is, add in the theo peak to the 3 way join
                join_pred_x_exp_theo[i] = join_pred_x_exp_pred[pos]
            except ValueError:
                pass
        # now look for missing prediction peaks and add them in
        for missing_pred in set(join_pred_x_exp_pred) - set(join_pred_x_exp_theo):
            join_exp_x_theo_exp.append(None)
            join_exp_x_theo_theo.append(None)
            join_pred_x_exp_theo.append(missing_pred)

        return join_exp_x_theo_exp, join_pred_x_exp_theo, join_exp_x_theo_theo

    @staticmethod
    def list2uint64(list_in):
        return pa.array(list_in, type=pa.uint64())

    @staticmethod
    def list2uint32(list_in):
        return pa.array(list_in, type=pa.uint32())

    @staticmethod
    def list2uint16(list_in):
        return pa.array(list_in, type=pa.uint16())

    @staticmethod
    def list2int16(list_in):
        return pa.array(list_in, type=pa.uint16())

    @staticmethod
    def list2float32(list_in):
        return pa.array(list_in, type=pa.float32())

    @staticmethod
    def list2float64(list_in):
        return pa.array(list_in, type=pa.float64())

    def to_pandas(self):
        """
        output the join results as a pandas dataframe
        """
        dtype_mapping = {
            pa.int8(): pd.Int8Dtype(),
            pa.int16(): pd.Int16Dtype(),
            pa.int32(): pd.Int32Dtype(),
            pa.int64(): pd.Int64Dtype(),
            pa.uint8(): pd.UInt8Dtype(),
            pa.uint16(): pd.UInt16Dtype(),
            pa.uint32(): pd.UInt32Dtype(),
            pa.uint64(): pd.UInt64Dtype(),
            pa.bool_(): pd.BooleanDtype(),
            pa.float32(): pd.Float32Dtype(),
            pa.float64(): pd.Float64Dtype(),
            pa.string(): pd.StringDtype(),
        }
        return self.results.to_pandas(types_mapper=dtype_mapping.get)


class PairwiseJoin(Join):
    """
    Join 2 sets of spectra
    """
    def __init__(self, exp_lib_map, theo_lib_map, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if issubclass(type(exp_lib_map), mss.HiResSpectrum) and issubclass(type(theo_lib_map), mss.HiResSpectrum):
            # allow for joining just two spectra
            self.exp_lib_map = msui.ListLibraryMap([exp_lib_map])
            self.theo_lib_map = msui.ListLibraryMap([theo_lib_map])
        else:
            self.exp_lib_map = exp_lib_map
            self.theo_lib_map = theo_lib_map

    def do_join(self, tiebreaker="mz"):
        experimental_list = []
        experimental_list_id = []
        theoretical_list = []
        theoretical_list_id = []

        def update_join_lists(result_in, exp_id, theo_id):
            experimental_list.extend(result_in[0])
            experimental_list_id.extend([exp_id] * len(result_in[0]))
            theoretical_list.extend(result_in[1])
            theoretical_list_id.extend([theo_id] * len(result_in[1]))

        assert(len(self.exp_lib_map) == len(self.theo_lib_map))

        for i in range(len(self.exp_lib_map)):
            result = self.join_2_spectra(self.exp_lib_map.getspectrum_by_row(i),
                                         self.theo_lib_map.getspectrum_by_row(i), tiebreaker=tiebreaker)
            update_join_lists(result, self.exp_lib_map.get_ids()[i], self.theo_lib_map.get_ids()[i])

        self.results = pa.Table.from_arrays([Join.list2uint64(experimental_list_id),
                                             Join.list2uint64(theoretical_list_id),
                                             Join.list2uint32(experimental_list),
                                             Join.list2uint32(theoretical_list)],
                                            names=['exp_id', 'theo_id', 'exp_peaks', 'theo_peaks'])
        return self


class ThreewayJoin(Join):
    """
    Join 3 sets of spectra. The join lists returned include all experimental and predicted peaks, but only the
    theoretical peaks that match the experimental spectra (and not the predicted spectrum).
    Note that it is possible to get a join where the theoretical peak matches the experimental peak but
    not the predicted peak.
    """
    def __init__(self, exp_lib_map, pred_lib_map, theo_lib_map, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if issubclass(type(exp_lib_map), mss.HiResSpectrum) and issubclass(type(theo_lib_map), mss.HiResSpectrum) and \
                issubclass(pred_lib_map, mss.HiResSpectrum):
            # allow for joining just two spectra
            self.exp_lib_map = msui.ListLibraryMap([exp_lib_map])
            self.pred_lib_map = msui.ListLibraryMap([pred_lib_map])
            self.theo_lib_map = msui.ListLibraryMap([theo_lib_map])
        else:
            self.exp_lib_map = exp_lib_map
            self.pred_lib_map = pred_lib_map
            self.theo_lib_map = theo_lib_map

    def do_join(self, tiebreaker="mz"):
        experimental_list = []
        theoretical_list = []
        predicted_list = []
        experimental_list_id = []
        predicted_list_id = []
        theoretical_list_id = []

        # per spectrum values
        ev = []
        precursor_mz = []
        cosine_score = []
        peptide_length = []
        charge = []
        num_pos_aa = []
        num_neg_aa = []
        ptm = []
        cterm_aa = []

        # per peak values
        annotations = []
        exp_mz = []
        exp_intensity = []
        pred_mz = []
        pred_intensity = []
        pred_stddev = []
        z_score = []
        theo_mz = []

        assert(len(self.exp_lib_map) == len(self.theo_lib_map) == len(self.pred_lib_map))

        for i in range(len(self.exp_lib_map)):
            experimental_spectrum = self.exp_lib_map.getspectrum_by_row(i).norm(max_intensity_in=1.0)
            predicted_spectrum = self.pred_lib_map.getspectrum_by_row(i).norm(max_intensity_in=1.0)
            theoretical_spectrum = self.theo_lib_map.getspectrum_by_row(i).norm(max_intensity_in=1.0)

            result = self.join_3_spectra(experimental_spectrum,
                                         predicted_spectrum,
                                         theoretical_spectrum, tiebreaker=tiebreaker)

            experimental_list.extend(result[0])
            predicted_list.extend(result[1])
            theoretical_list.extend(result[2])
            exp_id = self.exp_lib_map.get_ids()[i]
            if type(exp_id) == tuple:
                # deal with multiindex
                exp_id = exp_id[0]
            experimental_list_id.extend([exp_id] * len(result[0]))
            pred_id = self.pred_lib_map.get_ids()[i]
            if type(pred_id) == tuple:
                # deal with multiindex
                pred_id = pred_id[1]
            predicted_list_id.extend([pred_id] * len(result[1]))
            theo_id = self.theo_lib_map.get_ids()[i]
            if type(theo_id) == tuple:
                # deal with multiindex
                theo_id = theoretical_spectrum.id
            theoretical_list_id.extend([theo_id] * len(result[2]))

            # extract info from spectra
            exp_mz.append(pa.array(experimental_spectrum.products.mz,
                                   type=pa.float64()).take(pa.array(result[0], type=pa.int64())))
            exp_intensity.append(pa.array(experimental_spectrum.products.intensity,
                                          type=pa.float64()).take(pa.array(result[0], type=pa.int64())))
            pred_mz.append(pa.array(predicted_spectrum.products.mz,
                                    type=pa.float64()).take(pa.array(result[1], type=pa.int64())))
            pred_intensity.append(pa.array(predicted_spectrum.products.intensity,
                                           type=pa.float64()).take(pa.array(result[1], type=pa.int64())))
            theo_mz.append(pa.array(theoretical_spectrum.products.mz,
                                    type=pa.float64()).take(pa.array(result[2], type=pa.int64())))
            annotations.append(theoretical_spectrum.products.annotations.take(pa.array(result[2], type=pa.int64())))

            if predicted_spectrum.products.stddev is not None:
                pred_stddev.append(pa.array(predicted_spectrum.products.stddev,
                                            type=pa.float64()).take(pa.array(result[1], type=pa.int64())))
                # should use inf if divide by zero.  pyarrow supports inf, too.
                # z score is (observed - predicted/ predicted std dev
                z_score_subtract = pc.subtract(exp_intensity[-1], pred_intensity[-1])
                z_score_temp = pc.divide(z_score_subtract, pred_stddev[-1])
                # if the difference and stddev are both zero, such as for the base peak, set the z score to 0
                z_score_temp = pc.if_else(pc.and_(pc.equal(pred_stddev[-1], 0.0), pc.equal(z_score_subtract, 0.0)), 0.0, z_score_temp)
                z_score.append(z_score_temp)
                
                # z_score.append(pa.array(predicted_spectrum.products.intensity/predicted_spectrum.products.stddev,
                #                         type=pa.float64()).take(pa.array(result[1], type=pa.int64())))
            # per spectrum values
            row = self.exp_lib_map.getitem_by_row(i)
            if 'peptide' in row:
                peptide = row['peptide']
                peptide_length.extend([len(peptide)] * len(result[0]))
                num_neg_aa.extend([len(re.findall('[DE]', peptide))] * len(result[0]))
                num_pos_aa.extend([len(re.findall('[RHK]', peptide))] * len(result[0]))
                cterm_aa.extend([ord(peptide[-1]) - ord('A')] * len(result[0]))
                # annotate as phosphopeptide
                if 'mod_names' in row and 21 in row['mod_names']:
                    ptm.extend([21] * len(result[0]))
                else:
                    ptm.extend([0] * len(result[0]))

            def extend_with_dummy(array_in, lib_map, key, size):
                if key in lib_map:
                    array_in.extend([lib_map[key]] * size)

            extend_with_dummy(ev, row, 'ev', len(result[0]))
            extend_with_dummy(precursor_mz, row, 'precursor_mz', len(result[0]))
            extend_with_dummy(cosine_score, row, 'cosine_score', len(result[0]))
            extend_with_dummy(charge, row, 'charge', len(result[0]))

        exp_mz = pa.concat_arrays(exp_mz)
        exp_intensity = pa.concat_arrays(exp_intensity)
        pred_mz = pa.concat_arrays(pred_mz)
        pred_intensity = pa.concat_arrays(pred_intensity)
        theo_mz = pa.concat_arrays(theo_mz)
        annotations_structarray = pa.concat_arrays(annotations)

        arrays = [Join.list2uint64(experimental_list_id),
                  Join.list2uint64(predicted_list_id),
                  Join.list2uint64(theoretical_list_id),
                  Join.list2uint32(experimental_list),
                  Join.list2uint32(predicted_list),
                  Join.list2uint32(theoretical_list),
                  exp_mz,
                  exp_intensity,
                  pred_mz,
                  pred_intensity,
                  theo_mz,
                  ]
        names = ['exp_id', 'pred_id', 'theo_id', 'exp_peak_index', 'pred_peak_index', 'theo_peak_index',
                 'exp_mz', 'exp_intensity', 'pred_mz', 'pred_intensity', 'theo_mz']

        if pred_stddev:
            pred_stddev = pa.concat_arrays(pred_stddev)
            z_score = pa.concat_arrays(z_score)
            arrays += [pred_stddev, z_score]
            names += ['pred_stddev', 'z_score']
        if ev:
            arrays.append(Join.list2float32(ev))
            names.append('ev')
        if precursor_mz:
            arrays.append(Join.list2float64(precursor_mz))
            names.append('precursor_mz')
        if charge:
            arrays.append(Join.list2int16(charge))
            names.append('charge')
        if cosine_score:
            arrays.append(Join.list2float32(cosine_score))
            names.append('cosine_score')
        if peptide_length and num_neg_aa and num_pos_aa:
            arrays += [
                Join.list2uint16(peptide_length),
                Join.list2uint16(num_neg_aa),
                Join.list2uint16(num_pos_aa),
                pa.DictionaryArray.from_arrays(indices=cterm_aa, dictionary=pa.array(string.ascii_uppercase)),
                Join.list2int16(ptm)
                ]
            names += ['peptide_length', 'num_neg_aa', 'num_pos_aa', 'cterm_aa', 'ptm']
        if annotations:
            for i in range(len(annotations_structarray.type)):
                arrays.append(annotations_structarray.field(i))
                names.append(ion_annot_fields[i][0])

        new_table = pa.Table.from_arrays(arrays, names=names)
        # hack to avoid issue with nulls in categorical pandas column
        column_index = new_table.column_names.index('ion_subtype')
        new_table = new_table.set_column(column_index, 'ion_subtype', new_table[column_index].cast('string'))
        self.results = new_table
        return self
