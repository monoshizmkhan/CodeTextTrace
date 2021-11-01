class Evaluate():
    def __init__(self):
        self.actuals = []
        self.predictions = []
        self.confusion_matrix = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        self.recallScore = 0
        self.precisionScore = 0
        self.f1score = 0

    def updateAll(self, actuals, predictions):
        self.actuals = actuals
        self.predictions = predictions

    def update(self, actual, prediction):
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
        if denom == 0:
            res = 0
        else:
            res = self.confusion_matrix["TP"] / denom
        self.recallScore = res
        return res


    def precision(self):
        denom = self.confusion_matrix["TP"] + self.confusion_matrix["FP"]
        if denom == 0:
            res = 0
        else:
            res = self.confusion_matrix["TP"] / denom
        self.precisionScore = res
        return res


    def f1(self):
        recall = self.recall()
        precision = self.precision()
        self.recallScore = recall
        self.precisionScore = precision
        if recall + precision == 0:
            res = 0
        else:
            res = (2 * recall * precision) / (recall + precision)
        self.f1score = res
        return res


    def evaluate(self):
        self.calculate_confusion_matrix()
        self.f1()