import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def read_tecplot_block_dat(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    # --- Parse variable names ---
    var_line = next(l for l in lines if l.startswith("VARIABLES"))
    variables = re.findall(r'"([^"]+)"', var_line)

    # --- Parse zone size (I dimension) ---
    zone_line = next(l for l in lines if l.startswith("ZONE"))
    I = int(re.search(r'I\s*=\s*(\d+)', zone_line).group(1))

    nvar = len(variables)

    # --- Collect all numeric tokens after ZONE ---
    start_idx = lines.index(zone_line) + 1
    tokens = []
    for l in lines[start_idx:]:
        tokens.extend(l.split())

    tokens = iter(tokens)

    # --- Read block data ---
    data = {}
    for v in variables:
        vals = []
        while len(vals) < I:
            try:
                vals.append(float(next(tokens)))
            except StopIteration:
                raise RuntimeError("Unexpected end of file while reading data")
        data[v] = np.array(vals)

    return data


# Example usage
data = read_tecplot_block_dat("Data/Gargiulo2026/raw_01.dat")

print(data.keys())
print(data["R_middle"])



fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
ax.plot(data['X_mean'], data['Y_mean'], data['Z_mean'])
# ax.set_axis_off()
plt.show()
plt.close()