# rename old style hotm files to i2rgus style filenames
import glob
import pandas as pd
import os

for f in glob.glob("1*cx*png"):
    if "_" not in f:
        fsplit = f.split('.')
        camera = fsplit[1]
        product = fsplit[2]
        oldf = f
        newf = fsplit[0] + pd.Timestamp(int(fsplit[0]), unit='s').strftime('.%a.%b.%d_%H_%M_%S.GMT.%Y.unalakleet') + '.' + camera + '.' + product + '.png'
        print(oldf)
        print(newf)
        os.rename(oldf, newf)
