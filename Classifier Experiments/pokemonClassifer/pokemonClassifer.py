import pandas as pd
import numpy as np
import math

data_dir = "C:/Users/danlo/OneDrive/Desktop/Programming Projects/Machine-Learning-Experiments/Classifier Experiments/pokemonClassifer/datasets/"


#   Read in the pokemon datasets, 'simple' and 'extra',
#   containing simple info and extra info for each pokemon.
dfExtra  = pd.read_csv(data_dir + "pokemonExtraInfo.csv")
dfSimple = pd.read_csv(data_dir + "pokemonSimpleInfo.csv")


#   Check the column names and adjust them to overlap 
#   appropriately.
#print("dfSimple cols: ", list(dfSimple.columns))
dfExtra = dfExtra.drop( columns=['Unnamed: 0'] )
new_column_labels = ['#', 'Name', 'Type', 'Info', 'Abilities', 'Total', 'Weaknesses', 'Height', 'Weight']
dfExtra.columns = new_column_labels
#print("dfExtra cols : ", list(dfExtra.columns))


#   Adjust dfSimple to have collected rows for the same
#   name pokemon

#       First, adjust dfSimple to have the '#' column without 
#       leading 0s and space


for i in range(dfSimple.shape[0]):
    #print("rowBefore: ", dfSimple.iloc[i])
    #print("CurrentValue: ", dfSimple.iloc[i]['#'], "    newValue: ", math.floor(float(dfSimple.iloc[i]['#'])))
    dfSimple.iloc[i] = dfSimple.iloc[i].replace(to_replace=dfSimple.iloc[i]['#'], value=math.floor(float(dfSimple.iloc[i]['#'])))
    #print("rowAfter : ", dfSimple.iloc[i])
print("new values: ", dfSimple['#'])

unique_names = set(dfSimple['Name'])
#print(unique_names)

c_name = unique_names.pop()
typesDict = {}
while c_name:
    if c_name not in typesDict.keys():
        typesDict[c_name] = []

    rows = (dfSimple[dfSimple['Name']==c_name])
    #print( "Rows with ", c_name, " in : ",  rows)
    #print(len(rows.index))
    for row_ind in range(len(rows.index)):
        #print(rows.iloc[row_ind]['Type'])
        type = rows.iloc[row_ind]['Type']
        typesDict[c_name].append( type )
    #print(c_name, "  types: ", typesDict[c_name])
    try:
        c_name = unique_names.pop()
    except:
        c_name = None

dfSimple['Type'] = dfSimple.apply( lambda row: typesDict[row.Name],axis=1 )
#dfExtra['Type']  = dfExtra.apply( lambda row: typesDict[row.Name],axis=1 )

for name in typesDict.keys():
    row_inds = list(dfSimple[dfSimple['Name']==name].index)
    dfSimple.drop(row_inds[1:], inplace=True)

#   Inner join the datasets and check contents

print( "dfSimple shape: ", dfSimple.shape )
print( "dfExtra  shape: ", dfExtra.shape )

#dfExtra['#'] = dfExtra['#'].astype(str)
a = list(dfExtra['#'])[0]
b = list(dfSimple['#'])[0]
print(a)
print(b)
#print(type(a))
#print(type(b))


#   Merge the two datasets into one complete dataset
dfPDEX = dfSimple.merge( dfExtra, how='inner', on=['#', 'Name', 'Total'])
dfPDEX.drop( columns=['Type_y'], inplace=True )
new_column_labels = list(dfPDEX.columns)
new_column_labels[2] = 'Type'
dfPDEX.columns = new_column_labels


print( "dfPDEX shape  : ", dfPDEX.shape )
print( "dfPDEX columns: ", dfPDEX.columns )
print( "dfPDEX        : ", dfPDEX )


#   Now have complete dataset for the pokemon that uses both datasets




#### MACHINE LEARNING TIME ####


