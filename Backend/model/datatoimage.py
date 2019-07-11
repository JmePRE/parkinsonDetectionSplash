import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

filename = Path("PaHaW/PaHaW_public/")
for i in range(1, 99):
    f2o = filename / str(i).zfill(5) / (str(i).zfill(5)+"__1_1.svc")
    try:
        f = open(f2o, 'r')
    except FileNotFoundError:
        continue
    samples = int(f.readline())
    npa = np.zeros((10000, 10000, 3))
    x = []
    y = []
    for j in range(samples):
        instr = f.readline()
        xco = int(instr[0:4])
        yco = int(instr[5:9])
        npa[xco, yco] = [255, 255, 255]
        x.append(xco)
        y.append(yco)
    plt.plot(x, y)
    plt.savefig("training_data/"+str(i).zfill(5)+".png")
    plt.show()
