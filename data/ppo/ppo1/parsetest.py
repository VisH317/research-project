import csv

w = open("./dfake_proc.txt", "w")
writer = csv.writer(w)
f = open("./dfake.txt", "r")

disc_loss=0
wass_dist=0
gen_loss=0
val_loss=0

w = open("./testlog.csv".format(i+1), "w")
writer = csv.writer(w)
writer.writerow(["reward"])

for i in range(5):
    f = open("./testlog{num}.txt".format(i+1), "r")
    
    for line in f:
        if "Finished" in line:
            writer.writerow([float(line.split(":").strip())])
    
    f.close()

w.close()