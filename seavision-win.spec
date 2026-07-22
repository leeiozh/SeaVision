# -*- mode: python ; coding: utf-8 -*-
#
# ЕДИНАЯ штатная сборка для Windows 7 / 8.1 / 10 / 11 (БЕЗ matplotlib).
#
# Собирается на Python 3.8 + PyInstaller 5.13.2. Это «нижняя» точка, совместимая
# вверх: бинарники 3.8 запускаются на всех Windows от 7 до 11, тогда как сборка
# на 3.9+ на семёрке не стартует (Python 3.9+ импортирует из kernel32 функции,
# которых там нет: PssQuerySnapshot — Win8.1+, api-ms-win-core-path — Win10+;
# path-API лечится шимом, а PssQuerySnapshot — прямой импорт из KnownDLL и
# локально не подменяется). Поэтому единая сборка возможна только на 3.8.
#
# Следствия версии: PyInstaller 5 не знает параметра Analysis(optimize=...)
# (появился в 6), numpy/scipy закреплены последними cp38-колёсами
# (см. requirements-win.txt). Шим api-ms-win-core-path здесь НЕ нужен.

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
