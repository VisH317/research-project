import csv

w = open("./generatorv9.csv", "w")
writer = csv.writer(w)
f = open("./generatorv9.txt", "r")
i = 0

disc_loss=0
wass_dist=0
gen_loss=0
val_loss=0

writer.writerow(["disc_loss", "gen_loss", "wass_dist", "val_loss"])

for line in f:
    if line!="NOT RGB IMAGE":
        if i%7==2:
            disc_loss = round(float(line.split("[")[1].split("]")[0]), 4)
        elif i%7==3:
            wass_dist = round(float(line.split("[")[1].split("]")[0]), 4)
        elif i%7==4:
            gen_loss = round(float(line.split(": ")[1]), 4)
        elif i%7==5:
            val_loss = round(float(line.split(":")[1]), 4)
        elif i%7==0:
            disc_loss = 0
            gen_loss = 0
            val_loss = 0
        elif i%7==6:
            writer.writerow([disc_loss, gen_loss, wass_dist, val_loss])
        i+=1
