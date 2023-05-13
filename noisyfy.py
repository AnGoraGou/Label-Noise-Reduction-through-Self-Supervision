import numpy as np
import pandas as pd
import random

#read csv file
df = pd.read_csv('')

#get the label coloumn
#generate random ids to add label noise

train_root_dir = './subrat/Data/train_image/'
train_csv_path = './subrat/Data/filtered_320_solo_train.csv'

n = 0.4  #amount of noise to be added

df = pd.read_csv(train_csv_path)
print(df)


#The dx library currently enables DEX media type visualization of pandas DataFrames e.g. individual calls to dx.display()
list_unique = df.label.unique().tolist()
print(list_unique)
# exit()
for element in list_unique:
        print(f'the number of {element} is {len(df.loc[df.label == int(element)])}')



#  symetric noise of 20% between the below classes
normal_id = df.index[df.label == 0].tolist()
invasive_id = df.index[df.label== 3].tolist()
print(len(normal_id))

#symetric noise of 20% between the below classes
benign_id = df.index[df.label == 1].tolist()
inSitu_id = df.index[df.label == 2].tolist()
#print(len(benign_id))
#print(len(inSitu_id))


df_Noisy = df
print("Initialisation done for df_Noisy")

# select 20% of data from col1 column
Normal_noisy = int(len(normal_id) * n)
print(Normal_noisy)
Norm_noisy_id = random.sample(normal_id, Normal_noisy) # sample randomly from 20% of the list
Norm_noisy_id
# exit()

for idx in Norm_noisy_id:
        df_Noisy.at[idx, 'label'] = '3'
        print(f"{n*100}% of the Normal label has been converted to Invasive")


Invasive_noisy = int(len(invasive_id) * n)
print(Invasive_noisy)
# exit()
Inv_noisy_id = random.sample(invasive_id, Invasive_noisy) # sample randomly from 20% of the list


for idx in Inv_noisy_id:
        df_Noisy.at[idx, 'label'] = '0'
        print(f"{n*100}% of the Invasive label has been converted to Normal")



Benign_noisy = int(len(benign_id) * n)
print(Benign_noisy)
# exit()
Ben_noisy_id = random.sample(benign_id, Benign_noisy) # sample randomly from 20% of the list
Ben_noisy_id
# exit()

for idx in Ben_noisy_id:
        df_Noisy.at[idx, 'label'] = '2'
        print(f"{n*100}% of the Invasive label has been converted to Normal")



        InSitu_noisy = int(len(inSitu_id) * n)
        print(InSitu_noisy)


pd.write('')
