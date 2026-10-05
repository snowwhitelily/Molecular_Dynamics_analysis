import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

CSV = "results/ROS1/actout_ctlfit_fel_frame_points.csv"
OUT = "figures/ROS1/ros1_prepared_final/FEL_four_variants.png"
VARIANTS = [("WT","WT (reference)"), ("S1986F","S1986F"),
            ("Q2022P","Q2022P"), ("Q2022P_S1986F","Q2022P / S1986F")]
VMAX = 6.2  # kJ/mol

df = pd.read_csv(CSV)
# shared axes limits across the four
sub = df[df["mutant"].isin([v for v,_ in VARIANTS])]
xlim = (sub.PC1.min(), sub.PC1.max()); ylim = (sub.PC2.min(), sub.PC2.max())

def fel(pc1, pc2, bins=100):
    xc = np.linspace(*xlim, bins); yc = np.linspace(*ylim, bins)
    XX, YY = np.meshgrid(xc, yc)
    k = gaussian_kde(np.vstack([pc1, pc2]))
    d = k(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    d = np.clip(d/np.nanmax(d), 1e-10, 1.0)
    G = np.clip(-0.0083144626*300.0*np.log(d), 0, VMAX)
    return XX, YY, G

fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150)
im = None
for ax, (v, title) in zip(axes.ravel(), VARIANTS):
    m = df["mutant"] == v
    if m.sum() == 0:
        ax.set_title(f"{title}\n(not found)"); ax.axis("off"); continue
    XX, YY, G = fel(df.PC1[m].values, df.PC2[m].values)
    im = ax.pcolormesh(XX, YY, G, cmap="viridis_r", vmin=0, vmax=VMAX, shading="auto", rasterized=True)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_xlim(xlim); ax.set_ylim(ylim)

fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.25)
cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
fig.colorbar(im, cax=cax, label="Relative free energy (kJ/mol)")
fig.suptitle("Free energy landscapes — four variants (actout_ctlfit)",
             fontsize=13, fontweight="bold")
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("saved:", OUT)
