import numpy as np
import pandas as pd

df = pd.read_csv('../train_ver2.csv', nrows =500000, skipinitialspace=True)
#

#So now we get the count of nulls of each column
counter = df.isnull().sum()

#Then we get the percentage
counter_perc = (counter[::1]/df.shape[0]) *100
counter_perc = counter_perc.sort_values(ascending=False)


#sample = df.sample(30)

#Plot the only cols who have positive perc for nulls
x = counter_perc.loc[counter_perc > 0]
x.plot.bar()

#Remove the noisiest column
df = df.drop(['conyuemp', 'ult_fec_cli_1t'], axis=1)

#replace any missing, or corrupted values with nans(to be better suited for later replacement)
df= df.replace(to_replace = [' NA', 'NA', ' '], value = np.nan)

#Filter the data rows which have more than 75% of the features missing
#df = df.filter(lambda x: x[::2:.isnull().sum(axis=1) <

#Then we need to imputate the data
#Ofc we first need to perform some statstical analysis to know better what to actuall replace the nulls with
#df = df.fillna(0)

#Perform Feature scalling

#test the corr of the numeric cols against the products cols

# Visualize the categorical cols against the product cols