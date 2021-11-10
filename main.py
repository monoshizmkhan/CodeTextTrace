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
    train, test = dr.getExpData()
    #train, test = dr.getSample()
    #train, test = dr.getSample(a=100, b=20)
    #train, test = dr.getTrainTestData()
    m = dualModel(train[1:101], test[1:21])
    m.convertAndCompressData()
    m.train()
    m.test()
