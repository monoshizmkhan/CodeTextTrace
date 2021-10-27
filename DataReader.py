import pandas as pd
from pathlib import Path
import math

class DataReader():
    def __init__(self, projectTypes, filepaths):
        self.projectTypes = projectTypes
        self.filepaths = filepaths
        self.datas = {}
        self.commits = {}
        self.issues = {}
        self.fullData = pd.DataFrame(columns=["Issue", "Commit", "Link"])
        self.originalData = pd.DataFrame(columns=["Issue", "Commit", "Link"])
        self.processedData = pd.DataFrame(columns=["Issue", "Commit", "Link"])
        self.finalData = pd.DataFrame(columns=["Issue", "Commit", "Link"])
        self.trainingData = pd.DataFrame(columns=["Issue", "Commit", "Link"])
        self.testingData = pd.DataFrame(columns=["Issue", "Commit", "Link"])
        pd.set_option('display.max_columns', None)

    def readLinkFiles(self):
        print("Reading file with links")
        for i in range(len(self.filepaths)):
            self.datas[self.projectTypes[i]] = pd.read_csv(self.filepaths[i]+"clean_link.csv")

    def readCommitsIssues(self):
        for i in range(len(self.filepaths)):
            projectType = self.projectTypes[i]
            filepath = self.filepaths[i]
            print("Reading commits and issues for "+projectType+" projects.")
            ids = self.datas[projectType]
            commitFile = pd.read_csv(filepath+"clean_commit.csv")
            issueFile = pd.read_csv(filepath + "clean_issue.csv")
            for index1, row_id in ids.iterrows():
                for index2, row_commit in commitFile.iterrows():
                    if row_id["commit_id"]==row_commit["commit_id"]:
                        commit = row_commit["diff"]
                        for index3, row_issue in issueFile.iterrows():
                            if row_id["issue_id"]==row_issue["issue_id"]:
                                issue = row_issue["issue_comments"]
                                self.fullData = (self.fullData).append({"Issue":issue, "Commit":commit, "Link":1}, ignore_index=True)
        return self.fullData

    def saveOriginalData(self):
        print("Saving original data.")
        self.readLinkFiles()
        self.readCommitsIssues()
        if len(self.projectTypes)>1:
            (self.fullData).to_csv("original_data.csv")
        else:
            (self.fullData).to_csv("original_data"+self.projectTypes[0]+".csv")

    def processOriginalData(self):
        if len(self.projectTypes)>1:
            original_file = Path("original_data.csv")
        else:
            original_file = Path("original_data"+self.projectTypes[0]+".csv")
        if (original_file.is_file())==False:
            print("Original data doesn't exist.\nCreating original data from raw data.")
            self.saveOriginalData()
        print("Reading original data.")
        self.originalData = pd.read_csv(original_file)
        print("Shuffling data")
        self.originalData = self.originalData.sample(frac=1).reset_index(drop=True)
        print("Creating additional instances.")
        total = len(self.originalData.index)
        startFrom = 0
        fileNum=1
        for index1, row1 in (self.originalData).iterrows():
            if index1<startFrom:
                continue
            print(str(index1+1)+"/"+str(total))
            for index2, row2 in (self.originalData).iterrows():
                if index1==index2:
                    self.processedData = (self.processedData).append({"Issue":row1["Issue"], "Commit":row2["Commit"], "Link":1}, ignore_index=True)
                else:
                    self.processedData = (self.processedData).append({"Issue": row1["Issue"], "Commit": row2["Commit"], "Link": 0}, ignore_index=True)
            if (index1%200==0 and index1>0) or index1==(total-1):
                name = "processed_data"+str(fileNum)+".csv"
                if len(self.projectTypes)==1:
                    name = "processed_data"+self.projectTypes[0]+str(fileNum)+".csv"
                (self.processedData).to_csv(name)
                fileNum+=1
                self.processedData.drop(self.processedData.index, inplace=True)
        print("Data processed into different files.")


    def combineData(self):
        if len(self.projectTypes)==1:
            p = "processed_data"+self.projectTypes[0]+"1.csv"
        else:
            p = "processed_data1.csv"
        if (Path(p).is_file())==False:
            print("Divided data files do not exist.")
            print("Creating divided data files.")
            self.processOriginalData()
        for i in range(20):
            path = "processed_data"+self.projectTypes[0]+str(i+1)+".csv"
            if Path(path).is_file()==False:
                break
            fle = Path(path)
            print("Reading file "+path)
            temp = pd.read_csv(fle)
            self.finalData = pd.concat([self.finalData, temp], ignore_index=True)
        print("All data combined.")
        self.finalData = self.finalData.sample(frac=1).reset_index(drop=True)
        print("All data shuffled.")
        if len(self.projectTypes)>1:
            (self.finalData).to_csv("final_data.csv")
        else:
            (self.finalData).to_csv("final_data"+self.projectTypes[0]+".csv")
        print("Final data saved.")

    def readFinalData(self):
        if len(self.projectTypes)>1:
            p = "final_data.csv"
            if (Path(p).is_file())==False:
                self.combineData()
        else:
            p = "final_data"+self.projectTypes[0]+".csv"
            if (Path(p).is_file())==False:
                self.combineData()
        pd.set_option('display.max_columns', None)
        print("Reading entire data...")
        self.finalData = pd.read_csv(p)
        print("Entire data read.\n\n")

    def splitData(self):
        if len(self.projectTypes)>1:
            ptr = "training_data.csv"
            pte = "testing_data.csv"
        else:
            ptr = "training_data"+self.projectTypes[0]+".csv"
            pte = "testing_data"+self.projectTypes[0]+".csv"
        if (Path(ptr).is_file())==False and (Path(pte).is_file())==False:
            self.readFinalData()
            num_of_rows = len(self.finalData.index)
            print("Total data size: "+str(num_of_rows)+" rows")
            training_size = math.floor(num_of_rows*0.7)
            self.trainingData = self.finalData[0:training_size]
            print("Training set size: "+str(len(self.trainingData.index))+" rows")
            self.testingData = self.finalData[training_size:num_of_rows]
            print(self.trainingData.head())
            print("\n\n\n")
            print("Testing set size: " + str(len(self.testingData.index)) + " rows")
            print(self.testingData.head())
            print("\n\n\n")
            print("Saving data to csv files...")
            self.trainingData = self.trainingData.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
            self.testingData = self.testingData.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
            (self.trainingData).to_csv(ptr)
            (self.testingData).to_csv(pte)
            print("Data files saved.")
        else:
            print("Training and testing data already exist.")
            print("Reading data...")
            self.trainingData = pd.read_csv(ptr)
            self.testingData = pd.read_csv(pte)
            print("Training set size: " + str(len(self.trainingData.index)) + " rows\n")
            print(self.trainingData.head())
            print("\n\n\n")
            print("Testing set size: " + str(len(self.testingData.index)) + " rows\n")
            print(self.testingData.head())
            print("\n\n\n")

    def getTrainTestData(self):
        self.splitData()
        return self.trainingData, self.testingData

    def getSample(self):
        self.splitData()
        return self.trainingData[10:20], self.testingData[10:20]


