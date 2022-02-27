import os
from matplotlib import pyplot as plt
from matplotlib import style


langs = ["dk", "en", "ja", "fr"]
tasks = ["syn", "sem"]
models = ["d-bert"]


for lang in langs:
    for model in models:
        with open("./results/sent/" + lang + "_" + "syn" + "_" + model + ".txt") as f:
            
            epochs = []
            losses = []
            accuracies = []
            losses_test = []
            accuracies_validation = []
            
            i = 0
            for line in f:
                if "Epoch" in line:
                    l = line.split()
                    epochs.append(i)
                    losses.append(float(l[3].replace(',', '')))
                    accuracies.append(float(l[5]))
                    i += 1
                elif "Test" in line:
                    l = line.split()
                    losses_test.append(float(l[2]))
                    accuracies_validation.append(float(l[5]))
            
            plt.plot(epochs,accuracies_validation)

plt.legend(langs)
plt.ylim((0, 1.5))
plt.show()

