import csv

disc_loss=0
wass_dist=0
gen_loss=0
val_loss=0

w = open("./testlogparse.csv", "w")
writer = csv.writer(w)
writer.writerow(["reward"])

for i in range(6):
    f = open("./testlog{num}.txt".format(num=i+1), "r")
    
    for line in f:
        if "Finished" in line:
            writer.writerow([float(line.split(":")[1].strip())])
    
    f.close()

w.close()