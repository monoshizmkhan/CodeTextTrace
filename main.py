from DataReader import DataReader
from cosineModel import cosineModel
from nnModel import nnModel
from dualModels import dualModel

if __name__ == '__main__':
    #projectTypes = ["dbcii", "keras-team", "flask"]
    #filepaths = ["dbcli/pgcli/", "keras-team/keras/", "pallets/flask/"]
    projectTypes = ["dbcii"]
    filepaths = ["dbcli/pgcli/"]
    dr = DataReader(projectTypes, filepaths)
    #train, test = dr.getSample()
    train, test = dr.getTrainTestData()
    m = dualModel(train[1:10001], test[1:1001])
    m.convertAndCompressData()
    m.train()
    m.test()
    #m = nnModel(train[1:10001], test[1:1001])
    #m.createEmbeddingsCosine()
    #m = cosineModel(train[1:10001], test[1:1001])
    #m.createEmbeddingsCosine()
    #m.trainEmbeddingsCosine()
    #m.createEmbeddings()
    #m.cosine()
    #m.evaluate()
