import base64
import zlib
from collections import OrderedDict

import numpy as np

from .. import constants as _mkconstants
from ..utils import general as _mkgeneral


def spectra_to_msp(fp, spectra, annotate_peptide=False, ion_types=None):
    """
    write out an array-like of spectra  in msp format

    :param fp: stream or filename to write out to.  will append
    :param spectra: map containing the spectrum
    :param annotate_peptide: annotate the spectra as peptide
    :param ion_types: ion types for annotation
    """
    fp = _mkgeneral.open_if_filename(fp, 'w+')

    for i in range(len(spectra)):
        print(spectra[i].to_msp(annotate_peptide=annotate_peptide, ion_types=ion_types), file=fp)
    return


def spectra_to_mgf(fp, spectra, charge_list=None):
    """
    write out an array-like of spectra in mgf format

    :param fp: stream or filename to write out to.  will append
    :param spectra: name of the column containing the spectrum
    :param charge_list: list of charges for Mascot to search, otherwise use the CHARGE field
    """
    fp = _mkgeneral.open_if_filename(fp, 'w+')
    if charge_list is not None:
        charge_string = "CHARGE="
        for charge in charge_list:
            charge_sign = '+' if charge > 0 else '-'
            charge_string += str(charge) + charge_sign + ' '
        print(charge_string, file=fp)
 
    for i in range(len(spectra)):
        print("BEGIN IONS", file=fp)
        print(f"TITLE={spectra[i].name}", file=fp)
        # note that outputting the charge locks Mascot into searching only that charge
        if charge_list is None:
            charge_sign = '+' if spectra[i].charge > 0 else '-'
            print(f"CHARGE={spectra[i].charge}{charge_sign}", file=fp)
        if spectra[i].id is not None:
            print(f"SCANS={spectra[i].id}", file=fp)
        if spectra[i].precursor is not None:
            print(f"PEPMASS={spectra[i].precursor.mz} 999.0", file=fp)
        if spectra[i].retention_time is not None:
            print(f"RTINSECONDS={spectra[i].retention_time}", file=fp)
#        else:
#            print("RTINSECONDS=0.0", file=fp)
        for j in range(len(spectra[i].products.mz)):
            print(f'{spectra[i].products.mz[j]} {spectra[i].products.intensity[j]}', file=fp)
        print("END IONS\n", file=fp)
    return


def spectra_to_mzxml(fp, spectra, mzxml_attributes=None, min_intensity=_mkconstants.EPSILON, compress=True, use_id_as_scan=True):
    """
    write out an array-like of spectra in mzxml format

    :param fp: stream or filename to write out to.  will not append
    :param min_intensity: the minimum intensity value
    :param spectra: name of the column containing the spectrum
    :param mzxml_attributes: dict containing mzXML attributes
    :param use_id_as_scan: use spectrum.id instead of spectrum.scan
    :param compress: should the data be compressed?
    """

    fp = _mkgeneral.open_if_filename(fp, 'w')

    """
    Notes:
    - MaxQuant has particular requirements for the MZxml that it will accept.  For hints, see
      https://github.com/OpenMS/OpenMS/blob/develop/src/openms/source/FORMAT/HANDLERS/MzXMLHandler.cpp
      - maxquant does not allow empty spectra
      - depending on scan type, may have to force scan type to Full
      - may require lowMz, highMz, basePeakMz, basePeakIntensity, totIonCurrent
      - should have activationMethod=CID?
    - ideally use xmlschema, but using json input.  However, broken xml format for mzXML and maxquant argues for print()
    - http://sashimi.sourceforge.net/schema_revision/mzXML_2.1/Doc/mzXML_2.1_tutorial.pdf
    - https://www.researchgate.net/figure/An-example-mzXML-file-This-figure-was-created-based-on-the-downloaded-data-from_fig3_268397878
    - http://www.codems.de/reports/mzxml_for_maxquant/ issues:
      - must build record offset index at the end of the xml file
      - requires line breaks after attributes
      - manufacturer should be <msManufacturer category=”msManufacturer” value=”Thermo Finnigan” />
      - msResolution appears to be ignored by maxquant                    
    """

    # default values for attributes
    info = {'startTime': 0.0, 'endTime': 1000.0, 'scanCount': 0, 'fileName': 'unknown', 'fileType': 'RAWData',
            'fileSha1':  'fc6ffa16c1a8c2a4794d4fbb0b345d08e73fe577', 'msInstrumentID': '1',
            'msManufacturer': 'Thermo Scientific', 'msModel': 'Orbitrap Fusion Lumos', 'centroided': '1'
            }

    # overwrite default values
    if mzxml_attributes is not None:
        for k, v in mzxml_attributes.items():
            info[k] = v

    if info['scanCount'] == 0:
        for i in range(len(spectra)):
            # MQ doesn't like empty spectra
            if spectra[i].products is not None and len(spectra[i].products.mz) > 0:
                info['scanCount'] += 1

    # find the min max retention time
    min_retention_time = 1.0
    max_retention_time = 0.0
    for i in range(len(spectra)):
        if spectra[i].retention_time is not None:
            if spectra[i].retention_time < min_retention_time:
                min_retention_time = spectra[i].retention_time 
            if spectra[i].retention_time > max_retention_time:
                max_retention_time = spectra[i].retention_time 
    if min_retention_time <= max_retention_time:
        info["startTime"] = min_retention_time
        info["endTime"] = max_retention_time

    index = OrderedDict()

    # create the header
    print('<?xml version="1.0" encoding="ISO-8859-1"?>', file=fp)
    print('<mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2"', file=fp)
    print('       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"', file=fp)
    print('       xsi:schemaLocation="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2 '
          'http://sashimi.sourceforge.net/schema_revision/mzXML_3.2/mzXML_idx_3.2.xsd">', file=fp)
    # MaxQuant requires startTime and endTime
    print(f'  <msRun scanCount="{info["scanCount"]}" startTime="PT{info["startTime"]}S"'
        f' endTime="PT{info["endTime"]}S">', file=fp)

    # singleton children of msRun
    print(f'    <parentFile fileName="{info["fileName"]}"', file=fp)
    print(f'                fileType="{info["fileType"]}"', file=fp)
    print(f'                fileSha1="{info["fileSha1"]}"/>', file=fp)

    print(f'    <msInstrument msInstrumentID="{info["msInstrumentID"]}">', file=fp)
    # MaxQuant requires msManufacturer to be set to 'Thermo Scientific'
    print(f'      <msManufacturer category="msManufacturer" value="{info["msManufacturer"]}"/>', file=fp)
    print(f'      <msModel category="msModel" value="{info["msModel"]}"/>', file=fp)
    print(f'    </msInstrument>', file=fp)

    print(f'    <dataProcessing centroided="1">', file=fp)
    print(f'    </dataProcessing>', file=fp)

    # now the spectra
    scan_number = 1
    for i in range(len(spectra)):
        spectrum = spectra[i].filter(min_mz=1.0, min_intensity=min_intensity).norm(max_intensity_in=999.0)

        if spectrum.products is not None and len(spectrum.products.mz) > 0:
            if use_id_as_scan:
                scan = spectrum.id
            elif spectrum.scan is not None:
                scan = spectrum.scan
            else:
                scan = scan_number
                scan_number += 1

            index[scan] = fp.tell() + 4  # add 4 to get to first character of scan tag

            retentionTime = spectrum.retention_time if spectrum.retention_time is not None else 0.001*i
            collisionEnergy = spectrum.ev if spectrum.ev is not None else spectrum.nce if spectrum.nce is not None \
                else spectrum.collision_energy if spectrum.collision_energy is not None else ""
            if spectrum.ion_mode is None or spectrum.ion_mode == "P":
                polarity = "+"
            elif spectrum.ion_mode == "N":
                polarity = "-"

            print(f'    <scan num="{scan}"', file=fp)
            print('          scanType="Full"', file=fp)  # set to MS2
            print(f'          centroided="{info["centroided"]}"', file=fp)
            print('          msLevel="2"', file=fp)
            print(f'          peaksCount="{len(spectrum.products.mz)}"', file=fp)
            print(f'          polarity="{polarity}"', file=fp)
            print(f'          retentionTime="PT{retentionTime:.4f}S"', file=fp)
            if collisionEnergy != "":
                print(f'          collisionEnergy="{collisionEnergy:.4f}"', file=fp)
            print(f'          lowMz="{min(spectrum.products.mz):.4f}"', file=fp)
            print(f'          highMz="{max(spectrum.products.mz):.4f}"', file=fp)
            basePeak = np.argmax(spectrum.products.intensity)
            print(f'          basePeakMz="{spectrum.products.mz[basePeak]:.4f}"', file=fp)
            print(f'          basePeakIntensity="{spectrum.products.intensity[basePeak]:.4f}"', file=fp)
            print(f'          totIonCurrent="{spectrum.products.intensity.sum():.4f}"', file=fp)
            print(f'          msInstrumentID="1">', file=fp)

            precursorIntensity = spectrum.precursor.intensity if spectrum.precursor is not None and spectrum.precursor.intensity is not None else 999.0
            precursorMz = spectrum.precursor.mz if spectrum.precursor is not None else ""
            precursorCharge = spectrum.charge if spectrum.charge is not None else ""
            activationMethod = spectrum.instrument_type if spectrum.instrument_type is not None else "HCD"

            print(f'      <precursorMz precursorScanNum="{scan}" precursorIntensity="{precursorIntensity:.4f}"'
                  f' precursorCharge="{precursorCharge}" activationMethod="{activationMethod}">{precursorMz:.4f}</precursorMz>', file=fp)

            # create (mz, intensity) pairs
            data = np.ravel([spectrum.products.mz, spectrum.products.intensity], 'F')
            # convert to 32 bit floats, network byte order (big endian)
            data = np.ascontiguousarray(data, dtype='>f4')
            # zlib compress
            if compress:
                data = zlib.compress(data)
                compressed_len = len(data)
            # base64
            data = base64.b64encode(data)
            if compress:
                print('      <peaks compressionType="zlib"', file=fp)
                print(f'             compressedLen="{compressed_len}"', file=fp)
            else:
                print('      <peaks compressionType="none"', file=fp)
                print('             compressedLen="0"', file=fp)
            print('             precision="32"', file=fp)
            print('             byteOrder="network"', file=fp)
            print(f'             contentType="m/z-int">{data.decode("utf-8")}</peaks>', file=fp)  # or "mz-int"

            print('    </scan>', file=fp)

    print('  </msRun>', file=fp)
    indexOffset = fp.tell() + 2  # add 2 to get to first character of index
    print('  <index name="scan">', file=fp)
    for k, v in index.items():
        print(f'    <offset id="{k}">{v}</offset>', file=fp)
    print('  </index>', file=fp)
    print(f'  <indexOffset>{indexOffset}</indexOffset>', file=fp)
    print(f'  <sha1>{info["fileSha1"]}</sha1>', file=fp)
    print('</mzXML>', file=fp)

    return
