#include "ext.hpp"


/*
fingerprint_schema = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("fingerprint_rep", pa.binary()),
        pa.field("set", pa.dictionary(pa.int8(), pa.string()))
    ]
)

def bitvecToArrow(fp):
    return DataStructs.cDataStructs.BitVectToBinaryText(fp)

def arrowToBitvec(fp):
    return DataStructs.cDataStructs.CreateFromBinaryText(fp)

def fetch_mz(idx):
    x = pc.list_flatten(table.column("mz").slice(idx,1)).to_pandas()
    y = pc.list_flatten(table.column("intensity").slice(idx,1)).to_pandas()
    df = pd.DataFrame({"mz": x, "intensity": y})
    # Filter by minimum intensity
    df = df[df['intensity'] > df['intensity'].max()*MIN_INTENSITY_PERCENT]
    return df['mz'].to_numpy()

def fetch_spectrum(idx):
    x = pc.list_flatten(table.column("mz").slice(idx,1)).to_numpy()
    y = pc.list_flatten(table.column("intensity").slice(idx,1)).to_numpy()
    return spectrum.init_spectrum(True, mz=x, intensity=y)

def calc_fingerprint(mz):
    def calc_bit(a, b):
        return int(abs(b-a)*10)
    fp = DataStructs.ExplicitBitVect(20000)
    len_mz = len(mz)
    if USE_MZ_ZERO:
        for i in range(len_mz):
            #print(0, mz[i], calc_bit(0,mz[i]))
            fp.SetBit(calc_bit(0,mz[i]))
    for i in range(len_mz):
        for j in range(i+1,len_mz):
            #print(mz[i], mz[j], calc_bit(mz[i],mz[j]))
            fp.SetBit(calc_bit(mz[i],mz[j]))
    return fp

ids = []
fingerprints = []
tables = []
for i in range(len(table)):
    ids.append(table.column("id").slice(i,1).to_pandas()[0])
    mz = fetch_mz(i)
    fp = bitvecToArrow(calc_fingerprint(mz))
    fingerprints.append(fp)
    if (len(ids) >= 25000):
        tables.append(pa.table({"id": ids, "fingerprint_rep": fingerprints}, schema=fingerprint_schema))
        ids = []
        fingerprints = []
    #print(sys.getsizeof(fp))
    #print(DataStructs.cDataStructs.BitVectToText(fp))
if (len(ids) > 0):
    tables.append(pa.table({"id": ids, "fingerprint_rep": fingerprints}, schema=fingerprint_schema))
    ids = []
    fingerprints = []
fp_table = pa.concat_tables(tables)   
#fp_table = pa.table(
#    {
#        "id": table.column("id"),
#        "fingerprint": fingerprints
#    }, schema=fingerprint_schema)
#fp_table.to_pandas()
pq.write_table(fp_table, OUTFILENAME, row_group_size=250000, version="2.0")
fp_table.nbytes
*/