import pandas as pd 
import sys

filename = sys.argv[1]

with open(filename) as f:
    print(filename)
    data = pd.read_csv(filename)
    #print(data.columns)
    data["Match"] = data["true"] == data["preds"]
    # Here "positive" defined as the ungrammatical label; 1
    data["false_neg"] = data["preds"] < data["true"]
    data["false_pos"] = data["preds"] > data["true"]
    #print(data.head())
    #print(data.columns)
    data["true_pos"] = (data['true'] == 1) & (data['preds'] == 1)
    true_p = data['true_pos'].sum()
    false_p = data['false_pos'].sum()
    false_n = data['false_neg'].sum()
    precision = true_p/(true_p + false_p)
    recall = true_p/(true_p + false_n)
    F1 = 2*((precision * recall) / (precision + recall))
    print("False negatives:" + str(false_n))
    print("False positives:" + str(false_p))
    print("Precision:" + str(precision))
    print("Recall:" + str(recall))
    print("F1:" + str(F1))
    match = data["Match"].sum()
    acc = match / len(data.index)
    print("Accuracy:" + str(acc))