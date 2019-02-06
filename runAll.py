import scproj
import os

f=open("out.txt","w+")
for filename in os.listdir('data'):
	rezultat = scproj.start("data/"+str(filename))
	f.write(str(filename)+"\t"+str(rezultat)+"\n")
f.close()

