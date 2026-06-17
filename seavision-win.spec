# -*- mode: python ; coding: utf-8 -*-

excludes = [
    # GUI backends — не нужны
    'tkinter', '_tkinter', 'Tkinter',
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'wx', 'gi', 'gtk',
    # matplotlib — только для дебага (pics=false в штатной работе)
    'matplotlib',
    # NC/буй — нужны только для отладки (type=nc), в продакшне type=udp
    'netCDF4', 'cftime',
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
    datas=[],           # config.ini кладётся рядом с exe после установки
    hiddenimports=[
        'scipy.special.cython_special',
        'scipy.fft',
        'scipy.signal',
        'scipy.integrate',
        'scipy.ndimage',
        'scipy.optimize',
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
    exclude_binaries=True,   # onedir: dll-файлы лежат рядом, не внутри exe
    name='seavision',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch='x86_64',
    codesign_identity=None,
    entitlements_file=None,
    icon='sv.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='seavision',        # → dist/seavision/ — вся эта папка и есть дистрибутив
)
