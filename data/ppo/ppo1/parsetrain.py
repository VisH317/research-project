import csv

w = open("./dfake_proc.txt", "w")
writer = csv.writer(w)
f = open("./dfake.txt", "r")

disc_loss=0
wass_dist=0
gen_loss=0
val_loss=0

w = open("./log{num}.csv".format(i+1), "w")
writer = csv.writer(w)
writer.writerow(["ep_rew_mean", "loss"])

for i in range(5):
    f = open("./log{num}.txt".format(i+1), "r")
    erm = -1001
    loss = -1001
    
    for line in f:
        if "loss" in line:
            loss = float(line.split("|")[2].strip())
        elif "ep_rew_mean" in line:
            erm = float(line.split("|")[2].strip())
        if erm!=-1001 and loss!=-1001:
            writer.writerow([erm, loss])
            erm=-1001
            loss=-1001
    
    f.close()

w.close()