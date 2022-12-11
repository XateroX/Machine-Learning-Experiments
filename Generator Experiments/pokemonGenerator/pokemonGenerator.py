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


data_dir = "C:/Users/danlo/OneDrive/Desktop/Programming Projects/Machine-Learning-Experiments/Generator Experiments/pokemonGenerator/datasets/"


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
#       This will be done using another generator network
#       and will be trained on the success of 'fooling' the 
#       discriminator (the pokemon classifer)

class PokemonGenerator(tf.keras.Model):
    """Fully connected latent space -> pokemon dimensionality network."""

    def __init__(self, latent_dims):
        super(PokemonGenerator, self).__init__()
        self.latent_dims = latent_dims
        self.decoder = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(latent_dims,)),
                    tf.keras.layers.Dense(latent_dims*2)
                ]
        )
        self.statProcessing = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(6, activation=tf.keras.activations.sigmoid)
            ]
        )
        self.heightAndWeight = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid)
            ]
        )

        self.type1Output = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(19, activation=tf.keras.activations.softmax)
            ]
        )
        self.type2Output = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(19, activation=tf.keras.activations.softmax)
            ]
        )

        self.ability1Output = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(188, activation=tf.keras.activations.softmax)
            ]
        )
        self.ability2Output = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(188, activation=tf.keras.activations.softmax)
            ]
        )
        self.ability3Output = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(188, activation=tf.keras.activations.softmax)
            ]
        )

        self.info1Output = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(7284, activation=tf.keras.activations.softmax)
            ]
        )
        self.info2Output = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(7284, activation=tf.keras.activations.softmax)
            ]
        )

    def call(self, x):
        #   May need to process the output to be a reasonable set of values
        #   i.e. ints for the type selection, info and abilities
        raw_output = self.decoder(x)

        stats    = self.statProcessing(raw_output)

        heightAndWeight = self.heightAndWeight(raw_output)

        type1    = self.type1Output(raw_output)
        type2    = self.type2Output(raw_output)

        ability1 = self.ability1Output(raw_output)
        ability2 = self.ability2Output(raw_output)
        ability3 = self.ability3Output(raw_output)

        info1    = self.info1Output(raw_output)
        info2    = self.info2Output(raw_output)

        pokemon  = tf.concat( [stats, heightAndWeight, type1, type2, ability1, ability2, ability3, info1, info2], -1 )
        print("pokemon shape: ", pokemon.shape)
        return pokemon
    
    
    def processPokemon(self, raw):

        listVersion = []
        for val in raw[0]:
            listVersion.append(val.numpy())

        raw = listVersion


        completePokemon = [0.0]*15178

        completePokemon[0:6] = raw[0:6]

        completePokemon[6] = raw[6]*1000.0
        completePokemon[7] = raw[7]*1000.0

        completePokemon[ np.argmax( raw[8:27] ) ]       = 1.0
        completePokemon[ np.argmax( raw[27:46] ) ]      = 1.0

        completePokemon[ np.argmax( raw[46:234] ) ]     = 1.0
        completePokemon[ np.argmax( raw[234:422] ) ]    = 1.0
        completePokemon[ np.argmax( raw[422:610] ) ]    = 1.0

        completePokemon[ np.argmax( raw[610:7894] ) ]   = 1.0
        completePokemon[ np.argmax( raw[7894:15178] ) ] = 1.0



        return tf.convert_to_tensor( np.array(completePokemon), dtype=tf.float32)


#   Define the classifier model
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


#   Make the model to be trained to make pokemon
print("     Printing generator summary")
modelGEN = PokemonGenerator(100)
modelGEN.compile(optimizer = tf.keras.optimizers.Adam(1e-2), run_eagerly=True)#, loss=tf.keras.losses.CategoricalCrossentropy())
modelGEN(tf.random.uniform((1,modelGEN.latent_dims), minval=-1,maxval=1))
modelGEN.build((1,modelGEN.latent_dims))
modelGEN.decoder.summary()

print("     Printing classifier summary")
print("type MLFriendlyPokemonDataset: ", type(MLFriendlyPokemonDataset))
model = PokemonFCNN()
model.compile(optimizer = tf.keras.optimizers.Adam(1e-3), run_eagerly=True)#, loss=tf.keras.losses.CategoricalCrossentropy())
model(tf.convert_to_tensor( np.array(MLFriendlyPokemonDataset), dtype=tf.float32))
model.build((1,15178))
model.encoder.summary()
print("type MLFriendlyPokemonDataset: ", type(MLFriendlyPokemonDataset))

@tf.function
def generateRandomPokemon(modelGEN):
    randomLatentVector = tf.random.uniform((1,modelGEN.latent_dims), minval=-1,maxval=1)
    randomFakePokemon = modelGEN(randomLatentVector)
    return randomFakePokemon


#### TRAINING METHODS ####
@tf.function
def compute_loss(model, modelGEN, x, x_labels):
    data_r = x
    data_f = []
    for _ in range(len(x)):
        data_f.append( generateRandomPokemon(modelGEN) )
    data_f = tf.convert_to_tensor( data_f, dtype = tf.float32)
    data_f = tf.reshape( data_f, (data_f.shape[0],data_f.shape[2]) )

    #print( type(data_r) )
    #print( type(data_f) )
    print(data_r.shape)
    print(data_f.shape)

    d_r = model(data_r)
    d_f = model(data_f)

    return [-(tf.math.log(tf.reduce_mean(d_r))+tf.math.log(1-tf.reduce_mean(d_f))), (tf.math.log(1-tf.reduce_mean(d_f)))]

@tf.function
def train_step(model,modelGEN, x, x_labels):
	#Executes one training step and returns the loss.

	#This function computes the loss and gradients, and uses the latter to
	#update the model's parameters.

    with tf.GradientTape() as tape:
        tape.watch( modelGEN.trainable_variables )
        loss = compute_loss(model,modelGEN, x, x_labels)
    gradients_gen = tape.gradient(loss[1], modelGEN.trainable_variables)
    modelGEN.optimizer.apply_gradients(zip(gradients_gen, modelGEN.trainable_variables))
    with tf.GradientTape() as tape:
        tape.watch( model.trainable_variables )
        loss = compute_loss(model,modelGEN, x, x_labels)
    gradients_dis = tape.gradient(loss[0], model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients_dis, model.trainable_variables))



def test_accuracy(model, epoch, MLFriendlyPokemonDatasetTest):
    correct_preds = 0.0
    ds_subset = MLFriendlyPokemonDatasetTest.numpy()
    classification = model(ds_subset)#.numpy()
    for i in range(len(classification)):
        pred = (classification[i] > 0.5)[0]
        real = (labelsTest[i])[0]

        #print(pred)
        #print(real)

        if (pred and real == 1.0) or (not pred and real == 0.0):
            correct_preds += 1.0
    return correct_preds/len(MLFriendlyPokemonDatasetTest.numpy())




def train_accuracy(model, epoch, MLFriendlyPokemonDataset):
    correct_preds = 0.0
    ds_subset = MLFriendlyPokemonDataset.numpy()
    classification = model(ds_subset).numpy()
    for i in range(len(classification)):
        pred = (classification[i] > 0.5)[0]
        real = (labels[i])[0]

        #print("p:", pred)
        #print("r:", real)
        #print("p + r:", pred and real == 1.0)
        #print("!p + !r:", not pred and real == 0.0)

        if (pred and real == 1.0) or (not pred and real == 0.0):
            correct_preds += 1.0
    return correct_preds/len(MLFriendlyPokemonDataset.numpy())


MLFriendlyPokemonDataset_REAL = MLFriendlyPokemonDataset
print("type MLFriendlyPokemonDataset_REAL: ", type(MLFriendlyPokemonDataset_REAL))


fakePokemon = 700

#   The proportion to use as test data
p_test = 0.3
test_count = math.floor(p_test * (len(MLFriendlyPokemonDataset)+fakePokemon))
epochs = 200


train_accuracy_list = []
test_accuracy_list  = []
ganLoss             = []

generatedPokemonSeed = tf.random.uniform((1,modelGEN.latent_dims),minval=-1,maxval=1)

#   Epoch loop point...
for epoch in range(0,epochs + 1):
    print("epoch ", epoch, " starting")
    MLFriendlyPokemonDataset = []
    for i in range(len(MLFriendlyPokemonDataset_REAL)):
        MLFriendlyPokemonDataset.append(MLFriendlyPokemonDataset_REAL[i])
    labels = []
    for _ in range(len(MLFriendlyPokemonDataset)):
        labels.append([1.0])


    #for i in range(fakePokemon):
    #    randInd = random.randint(0,len(MLFriendlyPokemonDataset)-1)
    #    #fakePokemonExample = convertToMLFriendly( generateRandomPokemon(modelGEN) )
    #    fakePokemonExample = generateRandomPokemon(modelGEN).numpy()[0]
    #    #print("type fakePokemon: ", type(fakePokemonExample))
    #    #print(fakePokemonExample)
    #    MLFriendlyPokemonDataset.insert(randInd, fakePokemonExample)
    #    labels.insert(randInd, [0.0])

    #print("Generated fake pokemon list ")

    MLFriendlyPokemonDataset = np.array(MLFriendlyPokemonDataset)

    MLFriendlyPokemonDatasetTest = tf.convert_to_tensor( MLFriendlyPokemonDataset[:-test_count+1] , dtype=tf.float32)
    MLFriendlyPokemonDataset     = tf.convert_to_tensor( MLFriendlyPokemonDataset[test_count:]    , dtype=tf.float32)
    #print("Train Dataset shape: ", MLFriendlyPokemonDataset.shape)
    #print("Test Dataset shape : ", MLFriendlyPokemonDatasetTest.shape)

    labelsTest = labels[:-test_count+1]
    labels     = labels[  test_count: ]


    
    tf.keras.backend.set_value(model.optimizer.learning_rate,    1e-3*(1-min(epoch/epochs,0.6)))
    #tf.keras.backend.set_value(modelGEN.optimizer.learning_rate, 1e-3*(1-min(epoch/epochs,0.6)))

    classification = model(MLFriendlyPokemonDatasetTest.numpy())
    #print(classification.numpy())
    #print(np.array(labels))
    bce = tf.keras.losses.BinaryCrossentropy()
    #print(" 'loss': ", cce(labels, classification.numpy()))
    loss = bce(labelsTest, classification)

    start_time = time.time()
    train_step(model, modelGEN, MLFriendlyPokemonDataset.numpy(), labels)
    end_time = time.time()

    #cce = compute_loss(model, MLFriendlyPokemonDataset, labels)
    print('Epoch: {}, bce: {}, time: {}, train_acc {}, test_acc {}'
                .format(epoch, loss, round(end_time - start_time,3), round(train_accuracy(model,epoch, MLFriendlyPokemonDataset),3), round(test_accuracy(model,epoch, MLFriendlyPokemonDatasetTest),3)))
    print("Test generated pokemon: ", modelGEN(generatedPokemonSeed))
    generatorLoss = compute_loss(model,modelGEN, MLFriendlyPokemonDataset.numpy(), labels)[1]
    print("Generator Loss: ", generatorLoss)
    train_accuracy_list.append( train_accuracy(model,epoch, MLFriendlyPokemonDataset) )
    test_accuracy_list .append( test_accuracy(model,epoch, MLFriendlyPokemonDatasetTest) )
    ganLoss.append( generatorLoss )

    if epoch%10==0:
        accuracyDataFile = open(data_dir + "accuracy_plots/" + "rawdata.txt", 'w')
        for i in range(len(train_accuracy_list)):
            accuracyDataFile.write("tr " + str(i) + " " + str(train_accuracy_list[i]) + "\n")
            accuracyDataFile.write("te " + str(i) + " " + str(test_accuracy_list[i]) + "\n")
        accuracyDataFile.close()

        lossDataFile = open(data_dir + "loss_plots/" + "rawdata.txt", 'w')
        for i in range(len(ganLoss)):
            lossDataFile.write("ganloss " + str(i) + " " + str(ganLoss[i]) + "\n")
        lossDataFile.close()




'''
for i in range(round(len(MLFriendlyPokemonDataset)/10)):
    randInd = random.randint(0,len(MLFriendlyPokemonDataset)-1)
    randomPokemon = MLFriendlyPokemonDataset[randInd]
    verdict = "Fake"
    classification = model(MLFriendlyPokemonDataset.numpy())
    if classification[randInd] > 0.5:
        verdict = "Real"
    print(randomPokemon, " is ", verdict)
'''