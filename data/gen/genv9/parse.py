w = open("./dreal_proc.txt", "w")
f = open("./dreal.txt", "r")

disc_loss=0
wass_dist=0
gen_loss=0
val_loss=0


for line in f:
    w.write(line.split("[")[1].split("]")[0])
    w.write("\n")
