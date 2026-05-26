import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

df = pd.read_csv('batch_out/params.csv')

print(df)

# freqs = sorted(glob("batch_out/spec/*freqspec*"))
# dirs = sorted(glob("batch_out/spec/*direspec*"))
#
# for freq, dir in zip(freqs, dirs):
#     d_freq = np.load(freq)
#     plt.plot(d_freq)
#     plt.savefig(f"batch_out/pic/{freq.split('/')[-1][:-4]}.png")
#     plt.close()
#
#     d_dir = np.load(dir)
#     fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#     ax.set_theta_zero_location('N')
#     ax.set_theta_direction(-1)
#     ax.pcolormesh(np.linspace(0, 2 * np.pi, d_dir.shape[0]), np.arange(d_dir.shape[1]), d_dir.T)
#     plt.savefig(f"batch_out/pic/{dir.split('/')[-1][:-4]}.png")
#     plt.close()

print(df.columns)

plt.scatter(df["bdir"], df["d_p"])

plt.scatter(df["mdts"], df["d_p"])

mmin = min(df["bdir"].min(), df["bdir"].min())
mmax = max(df["bdir"].max(), df["bdir"].max())
plt.plot([mmin, mmax], [mmin, mmax])
plt.xlim([mmin, mmax])
plt.ylim([mmin, mmax])
plt.show()
