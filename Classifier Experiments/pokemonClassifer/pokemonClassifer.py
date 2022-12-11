import pandas as pd
import numpy as np
import math
import random
import time

#   ML imports
import tensorflow as tf
print("tensorflow version: ", tf.__version__)


cce = tf.keras.losses.CategoricalCrossentropy()

a = np.array( [[0.0,1.0],[1.0,0.0]] )
b = np.array( [[0.01,0.99],[0.01,0.99]] )

print(cce(a,b).numpy())


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
#print("new values: ", dfSimple['#'])

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
        c_type = rows.iloc[row_ind]['Type']
        typesDict[c_name].append( c_type )
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

#print( "dfSimple shape: ", dfSimple.shape )
#print( "dfExtra  shape: ", dfExtra.shape )


#   Merge the two datasets into one complete dataset
dfPDEX = dfSimple.merge( dfExtra, how='inner', on=['#', 'Name', 'Total'])
dfPDEX.drop( columns=['Type_y'], inplace=True )
new_column_labels = list(dfPDEX.columns)
new_column_labels[2] = 'Type'
dfPDEX.columns = new_column_labels


#print( "dfPDEX shape  : ", dfPDEX.shape )
#print( "dfPDEX columns: ", dfPDEX.columns )
#print( "dfPDEX        : ", dfPDEX )




#   Now have complete dataset for the pokemon that uses both datasets




#### MACHINE LEARNING TIME ####

#   15-vector for each pokemon uses indexs for worded properties,
#   these are the functions which convert words to their indices.


#   Type conversion to values
typeValDict = { 'NORMAL':0,
                    'FIGHTING':1,
                    'FLYING':2,
                    'POISON':3,
                    'GROUND':4,
                    'ROCK':5,
                    'BUG':6,
                    'GHOST':7,
                    'STEEL':8,
                    'FIRE':9,
                    'WATER':10,
                    'GRASS':11,
                    'ELECTRIC':12,
                    'PSYCHIC':13,
                    'ICE':14,
                    'DRAGON':15,
                    'DARK':16,
                    'FAIRY':17,
                    '':18}
def typeAsValue(c_type):
    return typeValDict[c_type]

def typesAsValues(typeList):
    returnTypes = []
    for c_type in typeList:
        returnTypes.append(typeAsValue(c_type))
    while len(returnTypes) < 2:
        returnTypes.append(18)
    return returnTypes


#   Abilities conversion to values
unique_abilities_init = list(dfPDEX['Abilities'])
unique_abilities = []
for i in range(len(unique_abilities_init)):
    #print(unique_abilities_init[i], len(unique_abilities_init[i]))
    unique_abilities_init[i] = unique_abilities_init[i][1:-2].replace(' ','').replace('\'', '').split(',')
    for ability in unique_abilities_init[i]:
        unique_abilities.append(ability)
unique_abilities = set(unique_abilities)
#print(unique_abilities, len(unique_abilities))
#print(len(set(unique_abilities)))

#print("MegaLauncher" in unique_abilities)

abilityValDict = {}
abilityValDict[''] = 0 
n = 1
for ability in unique_abilities:
    abilityValDict[ability] = n
    n+=1
#print(len(abilityValDict))

def abilityAsValue(ability):
    return abilityValDict[ability]
def abilitiesAsValues(abilities):
    returnList = []
    for ability in abilities:
        returnList.append(abilityValDict[ability])
    while len(returnList) < 3:
        returnList.append(0)
    return returnList



#   Info conversion values
file = open(data_dir+"nouns.txt", "r")
infoDict = {}
infoDict['']=0
n = 1
for line in file.readlines():
    infoDict[line.replace('\n', '')] = n
    n+=1
for existingInfo in dfPDEX['Info']:
    if existingInfo[:-8] not in infoDict.keys():
        infoDict[existingInfo[:-8]] = n
        n+=1
print("# of infos: ", len(infoDict))

def getInfoFromVal(val):
    for key in infoDict.keys():
        if infoDict[key] == val:
            return key
def infoAsValue(info):
    return infoDict[info]
def infosAsValues(infos):
    returnList = []
    for info in infos:
        returnList.append(infoAsValue(info))
    while len(returnList) < 2:
        returnList.append(0)
    return returnList

#   Dataset of 15-vectors that represent each pokemon in core values
pokemonDataset = []

#   Create pokemonDataset from dfPDEX
for i in range(dfPDEX.shape[0]):
    row = dfPDEX.iloc[i]
    #print(row)
    pokemonDataset.append( [float(row['HP']/200.0),
                            float(row['Attack']/200.0),
                            float(row['Defense']/200.0),
                            float(row['Special Attack']/200.0),
                            float(row['Special Defense']/200.0),
                            float(row['Speed']/200.0),
                            
                            float(row['Height']),
                            float(row['Weight']),
                            
                            typesAsValues(row['Type'])[0],
                            typesAsValues(row['Type'])[1],
                            
                            abilitiesAsValues(row['Abilities'][1:-2].replace(' ','').replace('\'', '').split(','))[0],
                            abilitiesAsValues(row['Abilities'][1:-2].replace(' ','').replace('\'', '').split(','))[1],
                            abilitiesAsValues(row['Abilities'][1:-2].replace(' ','').replace('\'', '').split(','))[2],
                            
                            infosAsValues([row['Info'][:-8]])[0],
                            infosAsValues([row['Info'][:-8]])[1]] )
print(np.array(pokemonDataset,dtype=object).shape)

randomPokemonInd = random.randint(0,dfPDEX.shape[0])
randomPokemon = pokemonDataset[randomPokemonInd]

print(dfPDEX.iloc[randomPokemonInd]['Name'], randomPokemon)
print("stats    :", randomPokemon[0:6])
print("type     :", list(typeValDict.keys())[randomPokemon[8]])
print("type     :", list(typeValDict.keys())[randomPokemon[9]])
print("ability1 :", list(abilityValDict.keys())[randomPokemon[10]])
print("ability2 :", list(abilityValDict.keys())[randomPokemon[11]])
print("ability3 :", list(abilityValDict.keys())[randomPokemon[12]])
print("info1    :", getInfoFromVal(randomPokemon[13]))
print("info2    :", getInfoFromVal(randomPokemon[14]))



#   Create a 'ML friendly' version of the 15-vector
def convertToMLFriendly(pokeVector):
    type1Extension    = [0]*19
    type2Extension    = [0]*19
    ability1Extension = [0]*188
    ability2Extension = [0]*188
    ability3Extension = [0]*188
    info1Extension    = [0]*7284
    info2Extension    = [0]*7284

    type1Extension[pokeVector[8]]     = 1
    type2Extension[pokeVector[9]]     = 1

    ability1Extension[pokeVector[10]] = 1
    ability2Extension[pokeVector[11]] = 1
    ability3Extension[pokeVector[12]] = 1

    info1Extension[pokeVector[13]]    = 1
    info2Extension[pokeVector[14]]    = 1

    pokeVector = pokeVector[0:8]
    pokeVector.extend(type1Extension)
    pokeVector.extend(type2Extension)

    pokeVector.extend(ability1Extension)
    pokeVector.extend(ability2Extension)
    pokeVector.extend(ability3Extension)

    pokeVector.extend(info1Extension)
    pokeVector.extend(info2Extension)

    return pokeVector

print(len(convertToMLFriendly(randomPokemon)))

MLFriendlyPokemonDataset = []
for pokemon in pokemonDataset:
    MLFriendlyPokemonDataset.append( convertToMLFriendly(pokemon) )
    #MLFriendlyPokemonDataset.append( pokemon )#convertToMLFriendly(pokemon) )

#   The labels of truth for the pokemon dataset
labels = []
for _ in range(len(MLFriendlyPokemonDataset)):
    labels.append([1.0])




#   Add 'fake' pokemon to the dataset

def generateRandomPokemon():
    randomRealPokemon = pokemonDataset[random.randint(0,len(pokemonDataset)-1)]

    randomRealPokemon[0] *= random.uniform(0.9,1.1)
    randomRealPokemon[1] *= random.uniform(0.9,1.1)
    randomRealPokemon[2] *= random.uniform(0.9,1.1)
    randomRealPokemon[3] *= random.uniform(0.9,1.1)
    randomRealPokemon[4] *= random.uniform(0.9,1.1)
    randomRealPokemon[5] *= random.uniform(0.9,1.1)

    randomRealPokemon[6] *= random.uniform(0.9,1.1)
    randomRealPokemon[7] *= random.uniform(0.9,1.1)

    randomRealPokemon[8] = random.randint(0,18)
    randomRealPokemon[9] = random.randint(0,18)

    randomRealPokemon[10] = random.randint(0,187)
    randomRealPokemon[11] = random.randint(0,187)
    randomRealPokemon[12] = random.randint(0,187)

    randomRealPokemon[13] = random.randint(0,7283)
    randomRealPokemon[14] = random.randint(0,7283)

    return randomRealPokemon

fakePokemon = 700
for i in range(fakePokemon):
    randInd = random.randint(0,len(MLFriendlyPokemonDataset)-1)
    fakePokemonExample = convertToMLFriendly( generateRandomPokemon() )
    #fakePokemonExample = generateRandomPokemon()#convertToMLFriendly( generateRandomPokemon() )
    MLFriendlyPokemonDataset.insert(randInd, fakePokemonExample )
    labels.insert(randInd, [0.0])

labels = np.array(labels)

#print(len(MLFriendlyPokemonDataset))
#print(len(labels))

#   Make the model
class PokemonFCNN(tf.keras.Model):
	"""Fully Connected Neural Network for pokemon classification."""

	def __init__(self):
		super(PokemonFCNN, self).__init__()
		self.encoder = tf.keras.Sequential(
				[
					tf.keras.layers.InputLayer(input_shape=(15178,)),
                    tf.keras.layers.Dense(10, activation=tf.keras.activations.sigmoid),
                    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid),
				]
		)

	def call(self, x):
		return self.encoder(x)

#   The proportion to use as test data
p_test = 0.3
test_count = math.floor(p_test * len(MLFriendlyPokemonDataset))

labelsTest = labels[:-test_count+1]
labels     = labels[  test_count: ]

MLFriendlyPokemonDatasetTest = tf.convert_to_tensor( np.array(MLFriendlyPokemonDataset[:-test_count+1]) , dtype=tf.float32)
MLFriendlyPokemonDataset     = tf.convert_to_tensor( np.array(MLFriendlyPokemonDataset[test_count:])    , dtype=tf.float32)
print( type(MLFriendlyPokemonDataset) )

model = PokemonFCNN()
model.compile(optimizer = tf.keras.optimizers.Adam(1e-2), run_eagerly=True)#, loss=tf.keras.losses.CategoricalCrossentropy())
model(MLFriendlyPokemonDataset)
model.build((1,15178))
model.encoder.summary()

#print(model( MLFriendlyPokemonDataset ))


epochs = 200


#### TRAINING METHODS ####

@tf.function
def compute_loss(model, x, x_labels):
    #x_labels = tf.convert_to_tensor( np.array(x_labels), dtype=tf.float32 )
    #x_labels = tf.reshape(x_labels, (x_labels.shape[0], 1))
    classification = model(x)
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(x_labels, classification)

@tf.function
def train_step(model, x, x_labels):
	#Executes one training step and returns the loss.

	#This function computes the loss and gradients, and uses the latter to
	#update the model's parameters.

    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, x_labels)
    #print([var.name for var in tape.watched_variables()])
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))



def test_accuracy(model, epoch):
    correct_preds = 0.0
    ds_subset = MLFriendlyPokemonDatasetTest.numpy()
    classification = model(ds_subset)#.numpy()
    for i in range(len(classification)):
        pred = classification[i] > 0.5
        real = labelsTest[i]

        #print(pred)
        #print(real)

        if (pred and real == 1) or (not pred and real == 0):
            correct_preds += 1.0
    return correct_preds/len(MLFriendlyPokemonDatasetTest.numpy())




def train_accuracy(model, epoch):
    correct_preds = 0.0
    ds_subset = MLFriendlyPokemonDataset.numpy()
    classification = model(ds_subset)#.numpy()
    for i in range(len(classification)):
        pred = classification[i] > 0.5
        real = labels[i]

        #print(pred)
        #print(real)

        if (pred and real == 1) or (not pred and real == 0):
            correct_preds += 1.0
    return correct_preds/len(MLFriendlyPokemonDataset.numpy())


#model.get_weights()
#print(model.get_weights()[0])

examplePokemon = tf.convert_to_tensor( np.array(pokemonDataset[0]) )
examplePokemon = tf.reshape( examplePokemon, (1, examplePokemon.shape[0]) )

for epoch in range(0,epochs + 1):
    
    tf.keras.backend.set_value(model.optimizer.learning_rate, 1e-2*(1-min(epoch/epochs,0.6)))

    classification = model(MLFriendlyPokemonDatasetTest.numpy())
    #print(classification.numpy())
    #print(np.array(labels))
    bce = tf.keras.losses.BinaryCrossentropy()
    #print(" 'loss': ", cce(labels, classification.numpy()))
    loss = bce(labelsTest, classification)

    start_time = time.time()
    train_step(model, MLFriendlyPokemonDataset.numpy(), labels)
    end_time = time.time()

    #cce = compute_loss(model, MLFriendlyPokemonDataset, labels)
    print('Epoch: {}, bce: {}, time: {}, train_acc {}, test_acc {}'
                .format(epoch, loss, round(end_time - start_time,3), round(train_accuracy(model,epoch),3), round(test_accuracy(model,epoch),3)))


for i in range(round(len(MLFriendlyPokemonDataset)/10)):
    randInd = random.randint(0,len(MLFriendlyPokemonDataset)-1)
    randomPokemon = MLFriendlyPokemonDataset[randInd]
    verdict = "Fake"
    classification = model(MLFriendlyPokemonDataset.numpy())
    if classification[randInd] > 0.5:
        verdict = "Real"
    print(randomPokemon, " is ", verdict)