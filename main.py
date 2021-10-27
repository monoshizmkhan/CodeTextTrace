from DataReader import DataReader
from Model import Model

if __name__ == '__main__':
    #projectTypes = ["dbcii", "keras-team", "flask"]
    #filepaths = ["dbcli/pgcli/", "keras-team/keras/", "pallets/flask/"]
    projectTypes = ["dbcii"]
    filepaths = ["dbcli/pgcli/"]
    dr = DataReader(projectTypes, filepaths)
    #train, test = dr.getSample()
    train, test = dr.getTrainTestData()
    m = Model(train[1:10001], test[1:1001])
    #m.createEmbeddingsCosine()
    m.trainEmbeddingsCosine()
    #m.createEmbeddings()
    #m.cosine()
    #m.evaluate()