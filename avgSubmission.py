import pandas as pd

submissionInput = list()
submissionOutput = list()
for i in range(1, 14):
    submissionInput.append(pd.read_csv('input/sub' + str(i) + '.csv'))
    id = submissionInput[0].id
    submissionOutput.append(pd.Series(submissionInput[i - 1].target))
finalSubmission = sum(submissionOutput) / 13.0
finalDF = pd.DataFrame({'id': id, 'target': finalSubmission})
finalDF.to_csv('output/avgSubmission.csv', index=False)
