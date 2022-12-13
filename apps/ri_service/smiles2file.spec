# -*- mode: python ; coding: utf-8 -*-
import os

# to run, 'pyinstaller --noconfirm smiles2file.spec'

block_cipher = None

# find conda lib
CONDA_PREFIX = os.environ['CONDA_PREFIX']

smiles2file_a = Analysis(['smiles2file.py'],
             pathex=['./nistms2'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['matplotlib', 'PIL', 'PyQt5', 'FixTk', 'tcl', 'tk', 'tkinter', 'Tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

shortest_paths_a = Analysis(['PA-Graph-Transformer/preprocess/shortest_paths.py'],
             pathex=['./PA-Graph-Transformer'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['matplotlib', 'PIL', 'PyQt5', 'FixTk', 'tcl', 'tk', 'tkinter', 'Tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

split_data_a = Analysis(['PA-Graph-Transformer/parse/split_data.py'],
             pathex=['./PA-Graph-Transformer'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['matplotlib', 'PIL', 'PyQt5', 'FixTk', 'tcl', 'tk', 'tkinter', 'Tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

train_prop_a = Analysis(['PA-Graph-Transformer/train/train_prop.py'],
             pathex=['./PA-Graph-Transformer'],
             binaries=[('model_99', '.'), (f'{CONDA_PREFIX}/lib/libiomp5.so', '.'), (f'{CONDA_PREFIX}/lib/libiomp5_db.so', '.'), (f'{CONDA_PREFIX}/lib/libiompstubs5.so', '.')],
             datas=[],
             hiddenimports=['scipy.special.cython_special'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['matplotlib', 'PIL', 'PyQt5', 'FixTk', 'tcl', 'tk', 'tkinter', 'Tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

MERGE( (smiles2file_a, 'smiles2file', 'smiles2file'), (shortest_paths_a, 'shortest_paths', 'shortest_paths'), (split_data_a, 'split_data', 'split_data'), (train_prop_a, 'train_prop', 'train_prop') )

smiles2file_pyz = PYZ(smiles2file_a.pure, smiles2file_a.zipped_data,
             cipher=block_cipher)
smiles2file_exe = EXE(smiles2file_pyz,
          smiles2file_a.scripts,
          [],
          exclude_binaries=True,
          name='smiles2file',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
smiles2file_coll = COLLECT(smiles2file_exe,
               smiles2file_a.binaries,
               smiles2file_a.zipfiles,
               smiles2file_a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='smiles2file')

shortest_paths_pyz = PYZ(shortest_paths_a.pure, shortest_paths_a.zipped_data,
             cipher=block_cipher)
shortest_paths_exe = EXE(shortest_paths_pyz,
          shortest_paths_a.scripts,
          [],
          exclude_binaries=True,
          name='shortest_paths',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
shortest_paths_coll = COLLECT(shortest_paths_exe,
               shortest_paths_a.binaries,
               shortest_paths_a.zipfiles,
               shortest_paths_a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='shortest_paths')

split_data_pyz = PYZ(split_data_a.pure, split_data_a.zipped_data,
             cipher=block_cipher)
split_data_exe = EXE(split_data_pyz,
          split_data_a.scripts,
          [],
          exclude_binaries=True,
          name='split_data',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
split_data_coll = COLLECT(split_data_exe,
               split_data_a.binaries,
               split_data_a.zipfiles,
               split_data_a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='split_data')

train_prop_pyz = PYZ(train_prop_a.pure, train_prop_a.zipped_data,
             cipher=block_cipher)
train_prop_exe = EXE(train_prop_pyz,
          train_prop_a.scripts,
          [],
          exclude_binaries=True,
          name='train_prop',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
train_prop_coll = COLLECT(train_prop_exe,
               train_prop_a.binaries,
               train_prop_a.zipfiles,
               train_prop_a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='train_prop')

# work around the bootloader looking in the wrong place for libpython
os.system(f"cp -rf {DISTPATH}/shortest_paths/* {DISTPATH}/smiles2file")
os.system(f"cp -rf {DISTPATH}/split_data/* {DISTPATH}/smiles2file")
os.system(f"cp -rf {DISTPATH}/train_prop/* {DISTPATH}/smiles2file")

