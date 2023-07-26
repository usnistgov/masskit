from masskit.utils.general import expand_path_list, parse_filename


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
