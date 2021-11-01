import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AutoTokenizer, AutoModel
import pandas as pd
from Evaluate import Evaluate


class dimensionNetwork(torch.nn.Module):
   def __init__(self):
      super().__init__()
      self.original_size = 768
      self.new_size = 16
      self.fc1 = torch.nn.Linear(self.original_size, self.new_size)
      self.fc2 = torch.nn.Linear(self.new_size, self.new_size)

   def forward(self, input):
      model = torch.nn.Sequential(self.fc1,
                                  torch.nn.Softmax(),
                                  self.fc2)
      output = model(input)
      return output

class classifierNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequenceLength = 16
        self.hiddenLength = 8
        self.batchLength = 8
        self.rnn_issue = torch.nn.RNN(self.sequenceLength, self.batchLength, self.sequenceLength)
        self.rnn_commit = torch.nn.RNN(self.sequenceLength, self.batchLength, self.sequenceLength)
        self.rnn_combined = torch.nn.RNN(self.sequenceLength, self.batchLength, self.sequenceLength)
        #self.rnn_combined = torch.nn.RNN(self.sequenceLength*2, self.batchLength, self.sequenceLength)
        self.classify = torch.nn.Softmax()

    def forward(self, input_issue, input_commit):
        h0 = torch.randn(1, self.batchLength, self.hiddenLength)
        output_issue, h_issue = self.rnn_issue(input_issue, h0)
        output_commit, h_commit = self.rnn_commit(input_commit, h0)
        #combined_input = torch.cat((output_issue, output_commit))
        #combined_hidden = torch.cat((h_issue, h_commit))
        combined_input = torch.mean(torch.stack([output_issue, output_commit]))
        combined_hidden = torch.mean(torch.stack([h_issue, h_commit]))
        rnns_output, final_hidden = self.rnn_combined(combined_input, combined_hidden)
        final_output = self.classify(rnns_output)
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

        self.dimensionNetwork = dimensionNetwork()

        self.classifierNetwork = classifierNetwork()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.classifierNetwork.parameters(), lr=0.001)
        self.epochs = 10

        self.inputs_i = []
        self.inputs_c = []
        self.inputs_labels = []

        self.test_i = []
        self.test_c = []
        self.test_labels = []

        self.ev = Evaluate()

    def convertAndCompressData(self):
        print("Creating training embeddings")
        train_len = len(self.tr.index)
        test_len = len(self.te.index)
        for index1, row1 in (self.tr).iterrows():
            if index1 % 100 == 0:
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

        print("\n\n\nCreating testing embeddings")
        for index1, row1 in (self.te).iterrows():
            if index1 % 100 == 0:
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

        print("\n\n\n")
        print("Train set size: " + str(len(self.inputs_labels)))
        print("Test set size: " + str(len(self.test_labels)))

    def train(self):
        print("Started training...\n")
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            outputs = (self.classifierNetwork()).forward(self.inputs_i, self.inputs_c)
            loss = self.criterion(outputs, self.inputs_labels)
            loss.backward()
            self.optimizer.step()
            epoch_loss=loss.item()
            print("Epoch: "+str(epoch)+", Loss: "+str(epoch_loss))
        print('\n\nFinished training\n\n')

    def test(self):
        print("Testing...\n")
        predictions = (self.classifierNetwork()).forward(self.test_i, self.test_c)
        actuals = self.test_labels
        self.ev.updateAll(actuals, predictions)
        print("\n\nConfusion matrix:")
        print(self.ev.confusion_matrix)
        print("\nRecall: "+str(self.ev.recallScore))
        print("Precision: "+str(self.ev.precisionScore))
        print("F1 score: "+str(self.ev.f1score))

