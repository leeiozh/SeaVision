# -*- mode: python ; coding: utf-8 -*-
#
# Debug-сборка: то же, что seavision-win.spec, но С matplotlib.
# Нужна, чтобы режим pics=<путь> (сохранение debug_combined.png) не падал
# из-за отсутствия matplotlib. Процессор рисует через Agg-backend
# (Figure + FigureCanvasAgg, без pyplot и без GUI), поэтому GUI-бэкенды
# по-прежнему исключены — тянем только сам matplotlib + Agg.

excludes = [
    # GUI backends — не нужны (рисуем в Agg → PNG)
    'tkinter', '_tkinter', 'Tkinter',
    'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'wx', 'gi', 'gtk',
    # matplotlib НЕ исключаем — он нужен для debug-картинок
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
        'scipy.signal',
        'scipy.ndimage',
        'scipy.interpolate',
        'netCDF4',
        'cftime',
        # matplotlib: только Agg-backend (savefig в PNG, без GUI)
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
    name='seavision',        # → dist-debug/seavision/ (workflow задаёт --distpath)
)
