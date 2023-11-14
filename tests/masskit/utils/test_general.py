import os
from masskit.utils.general import expand_path_list, get_file, parse_filename


def test_expand_path_list():
    path_list = ['~/*']
    path_list = expand_path_list(path_list)
    assert len(path_list) > 0

def test_parse_filename():
    root, suffix, compression = parse_filename('file.sdf')
    assert str(root) == 'file'
    assert suffix == 'sdf'
    assert compression == ''
    root, suffix, compression = parse_filename('file.sdf.gz')
    assert str(root) == 'file'
    assert suffix == 'sdf'
    assert compression == 'gz'

def test_get_file(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp('test_get_file')
    get_file('https://github.com/usnistgov/masskit_ai/releases/download/v.1.2.0/airi_model_v3_1.tgz', 
             cache_directory=tmpdir, search_path=[tmpdir], tgz_extension='.ckpt')
    assert os.path.getsize(tmpdir / 'airi_model_v3_1.ckpt') > 10000
    

