import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AutoTokenizer, AutoModel
import pandas as pd


class Model():
    def __init__(self, train, test):
        self.tr = train
        self.te = test
        self.trainSet = []
        self.testSet = []
        self.actuals = []
        self.predictions = []
        self.confusion_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.embModel = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.embModel.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.embModel = AutoModel.from_pretrained("microsoft/codebert-base")
        self.recallScore = 0
        self.precisionScore = 0
        self.F1score = 0
        print("Initialized")

    def trainEmbeddingsCosine(self):
        cos = torch.nn.CosineSimilarity(dim=0)
        threshold = 0.85
        alpha = 0.25
        trLen = len(self.tr.index)
        teLen = len(self.te.index)
        print("Training threshold")
        iterations = 5
        for itr in range(iterations):
            print("Iteration: "+str(itr+1))
            for index1, row1 in (self.tr).iterrows():
                if index1%100==0:
                    print("\nProgress: "+str(round(index1*100/trLen, 2))+"%, Threshold: "+str(threshold))
                i = row1["Issue"]
                c = row1["Commit"]
                link = row1["Link"]

                to_replace = ["\n", ",", ".", "?", "!", "<", ">", ";", ":", "(", ")", "{", "}", "[", "]", "/", "\\", "-",
                              "#", "`"]
                for ch in to_replace:
                    i = i.replace(ch, " ")
                while "  " in i:
                    i = i.replace("  ", " ")
                i_tokens = self.tokenizer.tokenize(i)
                i_tokens = [self.tokenizer.cls_token] + i_tokens + [self.tokenizer.sep_token]
                i_tokens = self.tokenizer.convert_tokens_to_ids(i_tokens)
                i_len = list((torch.tensor(i_tokens)).size())[0]
                if i_len > 512:
                    continue
                i_embeddings = self.embModel(torch.tensor(i_tokens)[None, :])[0]
                i_embeddings = torch.mean(i_embeddings, 0)
                i_embeddings = torch.mean(i_embeddings, 0)

                for ch in to_replace:
                    c = c.replace(ch, " ")
                while "  " in c:
                    c = c.replace("  ", " ")
                c_tokens = self.tokenizer.tokenize(c)
                c_tokens = [self.tokenizer.cls_token] + c_tokens + [self.tokenizer.sep_token]
                c_tokens = self.tokenizer.convert_tokens_to_ids(c_tokens)
                c_len = list((torch.tensor(c_tokens)).size())[0]
                if c_len > 512:
                    continue
                c_embeddings = self.embModel(torch.tensor(c_tokens)[None, :])[0]
                c_embeddings = torch.mean(c_embeddings, 0)
                c_embeddings = torch.mean(c_embeddings, 0)

                cos_score = cos(c_embeddings, i_embeddings)

                if cos_score>=threshold and link!=1:
                    #print("Old threshold: "+str(threshold)+", Cosine score: "+str(cos_score))
                    threshold = threshold + ((cos_score - threshold)*alpha)
                    #print("New threshold: "+str(threshold))
                elif cos_score<threshold and link==1:
                    #print("Old threshold: " + str(threshold) + ", Cosine score: " + str(cos_score))
                    threshold = threshold - ((threshold - cos_score)*alpha)
                    #print("New threshold: " + str(threshold))

        print("\n\n\nTesting")
        for index1, row1 in (self.te).iterrows():
            if index1%100==0:
                print("Progress: "+str(round(index1*100/teLen, 2))+"%")
                self.evaluate()
                print("Recall: "+str(round(self.recallScore, 2))+", Precision: "+str(round(self.precisionScore, 2))+", F-1 score: "+str(round(self.F1score, 2))+"\n")
            i = row1["Issue"]
            c = row1["Commit"]
            link = row1["Link"]

            to_replace = ["\n", ",", ".", "?", "!", "<", ">", ";", ":", "(", ")", "{", "}", "[", "]", "/", "\\", "-",
                          "#", "`"]
            for ch in to_replace:
                i = i.replace(ch, " ")
            while "  " in i:
                i = i.replace("  ", " ")
            i_tokens = self.tokenizer.tokenize(i)
            i_tokens = [self.tokenizer.cls_token] + i_tokens + [self.tokenizer.sep_token]
            i_tokens = self.tokenizer.convert_tokens_to_ids(i_tokens)
            i_len = list((torch.tensor(i_tokens)).size())[0]
            if i_len > 512:
                continue
            i_embeddings = self.embModel(torch.tensor(i_tokens)[None, :])[0]
            i_embeddings = torch.mean(i_embeddings, 0)
            i_embeddings = torch.mean(i_embeddings, 0)

            for ch in to_replace:
                c = c.replace(ch, " ")
            while "  " in c:
                c = c.replace("  ", " ")
            c_tokens = self.tokenizer.tokenize(c)
            c_tokens = [self.tokenizer.cls_token] + c_tokens + [self.tokenizer.sep_token]
            c_tokens = self.tokenizer.convert_tokens_to_ids(c_tokens)
            c_len = list((torch.tensor(c_tokens)).size())[0]
            if c_len > 512:
                continue
            c_embeddings = self.embModel(torch.tensor(c_tokens)[None, :])[0]
            c_embeddings = torch.mean(c_embeddings, 0)
            c_embeddings = torch.mean(c_embeddings, 0)

            prediction = 0
            actual = link
            cos_score = cos(c_embeddings, i_embeddings)
            if cos_score > threshold:
                prediction = 1
            self.actuals.append(actual)
            self.predictions.append(prediction)

        print("\n\n\nFinal results: \n")
        self.evaluate()
        print("Recall: " + str(round(self.recallScore, 2)) + ", Precision: " + str(
            round(self.precisionScore, 2)) + ", F-1 score: " + str(round(self.F1score, 2)) + "\n")
        outputFile = open("threshold_training_"+str(trLen)+"_"+str(teLen)+".txt", "w+")
        outputFile.write("Confusion matrix:\n")
        outputFile.write(str(self.confusion_matrix)+"\n")
        #for key, val in self.confusion_matrix:
            #outputFile.write(str(key)+": "+str(val)+"\n")
        outputFile.write("\nRecall: "+str(round(self.recallScore, 2)))
        outputFile.write("\nPrecision: " + str(round(self.precisionScore, 2)))
        outputFile.write("\nF-1 score: " + str(round(self.F1score, 2)))
        outputFile.close()


    def createEmbeddingsCosine(self):
        cos = torch.nn.CosineSimilarity(dim=0)
        print("Creating training embeddings and calculating cosine distance")
        threshold = 0.85
        printSteps = 20
        trLen = len(self.tr.index)
        teLen = len(self.te.index)
        totalLen = trLen + teLen
        outputFile = open("cosine.csv", "w+")
        outputFile.write("Instances, TP, FP, TN, FN, Recall, Precision, F1\n")
        print("Instances, TP, FP, TN, FN, Recall, Precision, F1")
        outputFile.close()
        for index1, row1 in (self.tr).iterrows():
            if index1 % printSteps == 0:
                print(str(index1)+"/"+str(totalLen)+", "+str(round(int(index1)*100/totalLen, 2))+"%")
                self.evaluate()
                print()
                toWrite = str(index1)+","+str(self.confusion_matrix["TP"])+","+str(self.confusion_matrix["FP"])+","+\
                          str(self.confusion_matrix["TN"])+","+str(self.confusion_matrix["FN"])+","+str(self.recallScore)\
                          +","+str(self.precisionScore)+","+str(self.F1score)+"\n"
                outputFile = open("cosine.csv", "a+")
                outputFile.write(toWrite)
                outputFile.close()
            i = row1["Issue"]
            c = row1["Commit"]
            link = row1["Link"]

            to_replace = ["\n", ",", ".", "?", "!", "<", ">", ";", ":", "(", ")", "{", "}", "[", "]", "/", "\\", "-",
                          "#", "`"]
            for ch in to_replace:
                i = i.replace(ch, " ")
            while "  " in i:
                i = i.replace("  ", " ")
            i_tokens = self.tokenizer.tokenize(i)
            i_tokens = [self.tokenizer.cls_token] + i_tokens + [self.tokenizer.sep_token]
            i_tokens = self.tokenizer.convert_tokens_to_ids(i_tokens)
            i_len = list((torch.tensor(i_tokens)).size())[0]
            if i_len > 512:
                continue
            i_embeddings = self.embModel(torch.tensor(i_tokens)[None, :])[0]
            i_embeddings = torch.mean(i_embeddings, 0)
            i_embeddings = torch.mean(i_embeddings, 0)

            for ch in to_replace:
                c = c.replace(ch, " ")
            while "  " in c:
                c = c.replace("  ", " ")
            c_tokens = self.tokenizer.tokenize(c)
            c_tokens = [self.tokenizer.cls_token] + c_tokens + [self.tokenizer.sep_token]
            c_tokens = self.tokenizer.convert_tokens_to_ids(c_tokens)
            c_len = list((torch.tensor(c_tokens)).size())[0]
            if c_len > 512:
                continue
            c_embeddings = self.embModel(torch.tensor(c_tokens)[None, :])[0]
            c_embeddings = torch.mean(c_embeddings, 0)
            c_embeddings = torch.mean(c_embeddings, 0)

            prediction = 0
            actual = link
            cos_score = cos(c_embeddings, i_embeddings)
            if cos_score > threshold:
                prediction = 1
            self.actuals.append(actual)
            self.predictions.append(prediction)

        print("\n\n\nCreating testing embeddings")
        for index1, row1 in (self.te).iterrows():
            if index1 % printSteps == 0:
                print(str(index1+trLen) + "/" + str(totalLen) + ", " + str(round(int(index1+trLen) * 100 / totalLen, 2)) + "%")
                self.evaluate()
                print()
                toWrite = str(index1+trLen) + "," + str(self.confusion_matrix["TP"]) + "," + str(
                    self.confusion_matrix["FP"]) + "," + \
                          str(self.confusion_matrix["TN"]) + "," + str(self.confusion_matrix["FN"]) + "," + str(
                    self.recallScore) \
                          + "," + str(self.precisionScore) + "," + str(self.F1score) + "\n"
                outputFile = open("cosine.csv", "a+")
                outputFile.write(toWrite)
                outputFile.close()
            i = row1["Issue"]
            c = row1["Commit"]
            link = row1["Link"]

            to_replace = ["\n", ",", ".", "?", "!", "<", ">", ";", ":", "(", ")", "{", "}", "[", "]", "/", "\\", "-",
                          "#", "`"]
            for ch in to_replace:
                i = i.replace(ch, " ")
            while "  " in i:
                i = i.replace("  ", " ")
            i_tokens = self.tokenizer.tokenize(i)
            i_tokens = [self.tokenizer.cls_token] + i_tokens + [self.tokenizer.sep_token]
            i_tokens = self.tokenizer.convert_tokens_to_ids(i_tokens)
            i_len = list((torch.tensor(i_tokens)).size())[0]
            if i_len > 512:
                continue
            i_embeddings = self.embModel(torch.tensor(i_tokens)[None, :])[0]
            i_embeddings = torch.mean(i_embeddings, 0)
            i_embeddings = torch.mean(i_embeddings, 0)

            for ch in to_replace:
                c = c.replace(ch, " ")
            while "  " in c:
                c = c.replace("  ", " ")
            c_tokens = self.tokenizer.tokenize(c)
            c_tokens = [self.tokenizer.cls_token] + c_tokens + [self.tokenizer.sep_token]
            c_tokens = self.tokenizer.convert_tokens_to_ids(c_tokens)
            c_len = list((torch.tensor(c_tokens)).size())[0]
            if c_len > 512:
                continue
            c_embeddings = self.embModel(torch.tensor(c_tokens)[None, :])[0]
            c_embeddings = torch.mean(c_embeddings, 0)
            c_embeddings = torch.mean(c_embeddings, 0)

            prediction = 0
            actual = link
            cos_score = cos(c_embeddings, i_embeddings)
            if cos_score > threshold:
                prediction = 1
            self.actuals.append(actual)
            self.predictions.append(prediction)

        print("\n\n\n")

    def createEmbeddings(self):
        print("Creating training embeddings")
        for index1, row1 in (self.tr).iterrows():
            if index1%20==0:
                print(str(index1)+"/"+str(len(self.tr.index)))
            i = row1["Issue"]
            c = row1["Commit"]
            link = row1["Link"]

            to_replace = ["\n", ",", ".", "?", "!", "<", ">", ";", ":", "(", ")", "{", "}", "[", "]", "/", "\\", "-",
                          "#", "`"]
            for ch in to_replace:
                i = i.replace(ch, " ")
            while "  " in i:
                i = i.replace("  ", " ")
            i_tokens = self.tokenizer.tokenize(i)
            i_tokens = [self.tokenizer.cls_token] + i_tokens + [self.tokenizer.sep_token]
            i_tokens = self.tokenizer.convert_tokens_to_ids(i_tokens)
            i_len = list((torch.tensor(i_tokens)).size())[0]
            if i_len>512:
                continue
            i_embeddings = self.embModel(torch.tensor(i_tokens)[None, :])[0]
            i_embeddings = torch.mean(i_embeddings, 0)
            i_embeddings = torch.mean(i_embeddings, 0)



            for ch in to_replace:
                c = c.replace(ch, " ")
            while "  " in c:
                c = c.replace("  ", " ")
            c_tokens = self.tokenizer.tokenize(c)
            c_tokens = [self.tokenizer.cls_token] + c_tokens + [self.tokenizer.sep_token]
            c_tokens = self.tokenizer.convert_tokens_to_ids(c_tokens)
            c_len = list((torch.tensor(c_tokens)).size())[0]
            if c_len>512:
                continue
            c_embeddings = self.embModel(torch.tensor(c_tokens)[None, :])[0]
            c_embeddings = torch.mean(c_embeddings, 0)
            c_embeddings = torch.mean(c_embeddings, 0)

            temp1 = {"Issue": i_embeddings, "Commit": c_embeddings, "Link": link}
            self.trainSet.append(temp1)

        print("\n\n\nCreating testing embeddings")
        for index1, row1 in (self.te).iterrows():
            if index1%20==0:
                print(str(index1)+"/"+str(len(self.te.index)))
            i = row1["Issue"]
            c = row1["Commit"]
            link = row1["Link"]

            to_replace = ["\n", ",", ".", "?", "!", "<", ">", ";", ":", "(", ")", "{", "}", "[", "]", "/", "\\", "-", "#", "`"]
            for ch in to_replace:
                i = i.replace(ch, " ")
            while "  " in i:
                i = i.replace("  ", " ")
            i_tokens = self.tokenizer.tokenize(i)
            i_tokens = [self.tokenizer.cls_token] + i_tokens + [self.tokenizer.sep_token]
            i_tokens = self.tokenizer.convert_tokens_to_ids(i_tokens)
            i_len = list((torch.tensor(i_tokens)).size())[0]
            if i_len > 512:
                continue
            i_embeddings = self.embModel(torch.tensor(i_tokens)[None, :])[0]
            i_embeddings = torch.mean(i_embeddings, 0)
            i_embeddings = torch.mean(i_embeddings, 0)

            for ch in to_replace:
                c = c.replace(ch, " ")
            while "  " in c:
                c = c.replace("  ", " ")
            c_tokens = self.tokenizer.tokenize(c)
            c_tokens = [self.tokenizer.cls_token] + c_tokens + [self.tokenizer.sep_token]
            c_tokens = self.tokenizer.convert_tokens_to_ids(c_tokens)
            c_len = list((torch.tensor(c_tokens)).size())[0]
            if c_len>512:
                continue
            c_embeddings = self.embModel(torch.tensor(c_tokens)[None, :])[0]
            c_embeddings = torch.mean(c_embeddings, 0)
            c_embeddings = torch.mean(c_embeddings, 0)

            temp2 = {"Issue": i_embeddings, "Commit": c_embeddings, "Link": link}
            self.testSet.append(temp2)


        print("\n\n\n")
        print("Train set size: "+str(len(self.trainSet)))
        print("Test set size: "+str(len(self.testSet)))

    def train(self):
        print("Do training.")

    def test(self):
        print("Do testing")

    def cosine(self):
        cos = torch.nn.CosineSimilarity(dim=0)
        for row in self.trainSet:
            i = row["Issue"]
            c = row["Commit"]
            actual = row["Link"]
            prediction = 0
            cos_score = cos(c, i)
            if cos_score>0.5:
                prediction = 1
            self.actuals.append(actual)
            self.predictions.append(prediction)
        for row in self.testSet:
            i = row["Issue"]
            c = row["Commit"]
            actual = row["Link"]
            prediction = 0
            cos_score = cos(c, i)
            if cos_score>0.5:
                prediction = 1
            self.actuals.append(actual)
            self.predictions.append(prediction)



    def calculate_confusion_matrix(self):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(self.actuals)):
            if self.actuals[i] == self.predictions[i] and self.actuals[i] == 1:
                TP += 1
            elif self.actuals[i] == self.predictions[i] and self.actuals[i] == 0:
                TN += 1
            elif self.actuals[i] != self.predictions[i] and self.actuals[i] == 1:
                FN += 1
            else:
                FP += 1
        self.confusion_matrix["TP"] = TP
        self.confusion_matrix["FP"] = FP
        self.confusion_matrix["TN"] = TN
        self.confusion_matrix["FN"] = FN

    def recall(self):
        denom = self.confusion_matrix["TP"] + self.confusion_matrix["FN"]
        if denom==0:
            res = 0
        else:
            res = self.confusion_matrix["TP"]/denom
        print("Recall: "+str(round(res, 3)))
        self.recallScore = res
        return res

    def precision(self):
        denom = self.confusion_matrix["TP"] + self.confusion_matrix["FP"]
        if denom == 0:
            res = 0
        else:
            res = self.confusion_matrix["TP"]/denom
        print("Precision: " + str(round(res, 3)))
        self.precisionScore = res
        return res

    def f1(self):
        recall = self.recall()
        precision = self.precision()
        self.recallScore = recall
        self.precisionScore = precision
        if recall+precision==0:
            res=0
        else:
            res = (2*recall*precision)/(recall+precision)
        print("F1 score: "+str(round(res, 3)))
        res = round(res, 3)
        self.F1score = res
        return res

    def evaluate(self):
        self.calculate_confusion_matrix()
        print(self.confusion_matrix)
        self.F1score = self.f1()