# -*- mode: python ; coding: utf-8 -*-

excludes = [
    # GUI backends — не нужны
    'tkinter', '_tkinter', 'Tkinter',
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'wx', 'gi', 'gtk',
    # matplotlib backends кроме agg
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_qt4agg',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_wxagg',
    'matplotlib.backends.backend_gtk3agg',
    'matplotlib.backends.backend_webagg',
    'matplotlib.backends._backend_tk',
    # тяжёлые пакеты, не нужны для main.py
    'pandas',
    'xarray',
    'ewdm',
    'pydap',
    'IPython', 'ipykernel', 'ipython_genutils',
    'jupyter', 'notebook', 'nbformat', 'nbconvert',
    'h5py',
    'setuptools', 'pkg_resources',
    'pytest',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[],           # config.ini остаётся рядом с exe
    hiddenimports=[
        'scipy.special.cython_special',
        'scipy.signal',
        'scipy.ndimage',
        'scipy.interpolate',
        'netCDF4',
        'cftime',
        'matplotlib.backends.backend_agg',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='seavision',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,        # strip не поддерживается на Windows
    upx=False,          # UPX может отсутствовать на сервере сборки
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='x86_64',
    codesign_identity=None,
    entitlements_file=None,
    icon='sv.ico',
)
