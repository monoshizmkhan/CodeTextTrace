import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AutoTokenizer, AutoModel
import gc
import pandas as pd
from Evaluate import Evaluate
import math
import time


class dimensionNetwork(torch.nn.Module):
   def __init__(self, original_size = 768, new_size=16):
      super().__init__()
      self.original_size = original_size
      self.new_size = new_size
      self.fc1 = torch.nn.Linear(self.original_size, self.new_size)
      self.model = torch.nn.Sequential(self.fc1, torch.nn.Softmax())

   def forward(self, input):
      output = self.model(input)
      return output

class classifierNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequenceLength = 16
        self.hiddenLength = 8
        self.batchLength = 8
        self.rnn_issue = torch.nn.RNN(self.sequenceLength, self.batchLength, self.sequenceLength)
        self.rnn_commit = torch.nn.RNN(self.sequenceLength, self.batchLength, self.sequenceLength)
        self.rnn_combined = torch.nn.RNN(int(self.sequenceLength/2), self.batchLength, int(self.sequenceLength/2))
        self.classify = dimensionNetwork(original_size=int(self.sequenceLength/2), new_size=2)

    def forward(self, input_issue, input_commit):
        h0 = torch.randn(self.sequenceLength, 1, self.hiddenLength)

        output_issue, h_issue = self.rnn_issue(input_issue, h0)
        output_commit, h_commit = self.rnn_commit(input_commit, h0)

        combined_input = torch.mean(torch.stack([output_issue, output_commit]), 0, keepdim=True)
        combined_input = torch.squeeze(combined_input, 0)

        combined_hidden = torch.mean(torch.stack([h_issue, h_commit]), 0, keepdim=True)
        combined_hidden = combined_hidden.detach().clone()
        combined_hidden.resize_(self.hiddenLength, 1, self.hiddenLength)

        rnns_output, final_hidden = self.rnn_combined(combined_input, combined_hidden)
        rnns_output = torch.squeeze(rnns_output)

        final_output = self.classify.forward(rnns_output)
        return final_output


class dualModel():
    def __init__(self, train, test):
        self.tr = train
        self.te = test

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.embModel = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.embModel.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.embModel = AutoModel.from_pretrained("microsoft/codebert-base")

        self.reducedSize = 16
        self.dimensionNetwork = dimensionNetwork(original_size=768, new_size=self.reducedSize)

        self.classifierNetwork = classifierNetwork()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.classifierNetwork.parameters(), lr=0.001)
        self.epochs = 3

        self.inputs_i = []
        self.inputs_c = []
        self.inputs_labels = []

        self.test_i = []
        self.test_c = []
        self.test_labels = []

        self.ev = Evaluate()

    def convertAndCompressDataDivided(self):
        print("Creating training embeddings\n")
        starting_time = time.time()
        tr = pd.read_csv("original_exp_training_0.csv")
        ln = len(tr.index)*10
        test_len = len(self.te.index)
        for part in range(10):
            print("\nPart-"+str(part+1))
            tr = pd.read_csv("original_exp_training_"+str(part)+".csv")
            for index1, row1 in (tr).iterrows():
                if index1 % 10 == 0:
                    current_time = time.time()
                    diff = round(current_time - starting_time, 3)
                    print("Total progress: "+str(index1+(185*part)) + "/" + str(ln)+", "+str(round((index1+(185*part))*100/ln, 2))+"%, Time difference: "+str(diff)+"s")
                    starting_time = time.time()

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

                c_embeddings = self.dimensionNetwork.forward(c_embeddings)
                i_embeddings = self.dimensionNetwork.forward(i_embeddings)

                self.inputs_c.append(c_embeddings)
                self.inputs_i.append(i_embeddings)
                self.inputs_labels.append(link)

                del c_embeddings
                del i_embeddings
                c_embeddings = torch.empty(1, self.reducedSize)
                i_embeddings = torch.empty(1, self.reducedSize)
                gc.collect()

        print("\n\n\nCreating testing embeddings")
        for index1, row1 in (self.te).iterrows():
            if index1 % 10 == 0:
                print(str(index1)+"/"+str(test_len)+", "+str(round(index1*100/test_len, 2))+"%")

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

            c_embeddings = self.dimensionNetwork.forward(c_embeddings)
            i_embeddings = self.dimensionNetwork.forward(i_embeddings)

            self.test_c.append(c_embeddings)
            self.test_i.append(i_embeddings)
            self.test_labels.append(link)

            del c_embeddings
            del i_embeddings
            c_embeddings = torch.empty(1, self.reducedSize)
            i_embeddings = torch.empty(1, self.reducedSize)
            gc.collect()

        del self.tr
        del self.te
        gc.collect()
        self.tr = pd.DataFrame()
        self.te = pd.DataFrame()
        print("\n\n\n")
        print("Train set size: " + str(len(self.inputs_labels)))
        print("Test set size: " + str(len(self.test_labels)))

    def convertAndCompressData(self):
        print("Creating training embeddings")
        train_len = len(self.tr.index)
        test_len = len(self.te.index)
        for index1, row1 in (self.tr).iterrows():
            if index1 % 10 == 0:
                print(str(index1) + "/" + str(train_len)+", "+str(round(index1*100/train_len, 2))+"%")

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

            c_embeddings = self.dimensionNetwork.forward(c_embeddings)
            i_embeddings = self.dimensionNetwork.forward(i_embeddings)

            self.inputs_c.append(c_embeddings)
            self.inputs_i.append(i_embeddings)
            self.inputs_labels.append(link)

            del c_embeddings
            del i_embeddings
            c_embeddings = torch.empty(1, self.reducedSize)
            i_embeddings = torch.empty(1, self.reducedSize)
            gc.collect()

        print("\n\n\nCreating testing embeddings")
        for index1, row1 in (self.te).iterrows():
            if index1 % 10 == 0:
                print(str(index1)+"/"+str(test_len)+", "+str(round(index1*100/test_len, 2))+"%")

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

            c_embeddings = self.dimensionNetwork.forward(c_embeddings)
            i_embeddings = self.dimensionNetwork.forward(i_embeddings)

            self.test_c.append(c_embeddings)
            self.test_i.append(i_embeddings)
            self.test_labels.append(link)

            del c_embeddings
            del i_embeddings
            c_embeddings = torch.empty(1, self.reducedSize)
            i_embeddings = torch.empty(1, self.reducedSize)
            gc.collect()

        del self.tr
        del self.te
        gc.collect()
        self.tr = pd.DataFrame()
        self.te = pd.DataFrame()
        print("\n\n\n")
        print("Train set size: " + str(len(self.inputs_labels)))
        print("Test set size: " + str(len(self.test_labels)))

    def train(self):
        print("Started training...\n")
        #labels_tensors = torch.Tensor(self.inputs_labels)
        lbl = torch.tensor([self.inputs_labels])
        lbl = torch.transpose(lbl, 0, 1)
        lbl = torch.squeeze(lbl)
        outputs = torch.empty(1, 2)
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            itr = len(self.inputs_i)
            for i in range(itr):
                output = (self.classifierNetwork).forward(torch.unsqueeze(torch.unsqueeze(self.inputs_i[i], 0), 0), torch.unsqueeze(torch.unsqueeze(self.inputs_c[i], 0), 0))
                output = torch.squeeze(output)
                output = torch.unsqueeze(output, dim=0)
                if i==0:
                    outputs = output
                else:
                    outputs = torch.cat((outputs, output), 0)
            loss = self.criterion(outputs, lbl)
            loss.backward(retain_graph=True)
            self.optimizer.step()
            epoch_loss=loss.item()
            print("Epoch: "+str(epoch)+", Loss: "+str(epoch_loss))
        print('\n\nFinished training\n\n')

    def test(self):
        print("Testing...\n")
        predictions = []
        for i in range(len(self.test_i)):
            prediction = ((self.classifierNetwork).forward(torch.unsqueeze(torch.unsqueeze(self.test_i[i], 0), 0), torch.unsqueeze(torch.unsqueeze(self.test_c[i], 0), 0)))
            prediction = prediction.tolist()
            prediction = prediction.index(max(prediction))
            predictions.append(prediction)
        actuals = self.test_labels
        self.ev.updateAll(actuals, predictions)
        self.ev.calculate_confusion_matrix()
        print("\n\nConfusion matrix:")
        print(self.ev.confusion_matrix)
        print("\nRecall: "+str(self.ev.recallScore))
        print("Precision: "+str(self.ev.precisionScore))
        print("F1 score: "+str(self.ev.f1score))

