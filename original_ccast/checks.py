import numpy as np, pandas as pd, glob

print("===== CHECK 1: are gas and hard coal identical per seed? =====")
for f in sorted(glob.glob("results/comparison/dispatch_ICO_seed*.npy")):
    D = np.load(f)
    print(f.split("/")[-1],
          "gas =", round(D[:, 2].mean()),
          " hardcoal =", round(D[:, 3].mean()))

print("\n===== CHECK 2: is the gas ceiling a real peak or an outlier? =====")
de = pd.read_csv("data/modified_dataset.csv")
tr = int(len(de) * 0.8)
g = de["generation fossil gas"].iloc[:tr]
print("mean  ", round(g.mean()))
print("max   ", round(g.max()), "  <- this sets the 20,034 ceiling")
print("99th  ", round(g.quantile(0.99)))
print("99.9th", round(g.quantile(0.999)))
