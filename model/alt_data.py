import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
filename = Path("PaHaW/PaHaW_public/")
tagfile = filename / "tagsfile.csv"
pcounter = 0 #//0-29
hcounter = 0 #//0-30
try:
    tfile = open(tagfile, 'r', encoding="utf-8-sig")
except FileNotFoundError:
    print("die")
for i in range(1, 99):
    print("i=", i)
    op = 0
    savepath = ["t_data1/healthy/", "t_data1/parkinson/", "testing/healthy/", "testing/parkinson/"]
    f2o = filename / str(i).zfill(5) / (str(i).zfill(5)+"__1_1.svc")
    try:
        f = open(f2o, 'r')
        tag = tfile.readline().rstrip('\n').split(",")
    except FileNotFoundError:
        print("fnf")
        continue

    if int(tag[0]) != i:
        print("tagmismatch", int(tag[0]))
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
    plt.rcParams["figure.figsize"] = (5, 5)
    plt.plot(x, y)
    plt.savefig("training_data/"+str(i).zfill(5)+".png")
    plt.clf()
    im = Image.open("training_data/"+str(i).zfill(5)+".png").convert("L")
    print(im.format, im.size, im.mode, i)
    im = im.crop((70, 65, 440, 445))
    print(tag[1])
    if tag[1] == 'H':
        hcounter += 1
        op = 0
        print("healthy, train:", savepath[op])
    elif tag[1] == 'P':
        pcounter += 1
        op = 1
        print("parkinson's, train:", savepath[op])
    im.save(savepath[op]+str(i).zfill(5)+"x.png", "PNG")
