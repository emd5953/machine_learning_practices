import numpy as np
import math
import sys

# Gaussian probability density function
def gaussian_pdf(x, mean, std):
    if std == 0:
        return 1.0 if x == mean else 0.0
    exponent = math.exp(- ((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent

def run_naive_bayes(trainingFile, testingFile):
    Xtrain = np.loadtxt(trainingFile)
    n = Xtrain.shape[0]
    d = Xtrain.shape[1] - 1 
    print("Training data:", n, "samples with", d, "features")

    features_train = Xtrain[:, :d]
    labels_train = Xtrain[:, d]

    class_pos = features_train[labels_train == 1]
    class_neg = features_train[labels_train == -1]

    mean_pos = np.mean(class_pos, axis=0)
    std_pos = np.std(class_pos, axis=0, ddof=1) 
    mean_neg = np.mean(class_neg, axis=0)
    std_neg = np.std(class_neg, axis=0, ddof=1)

    # prior probabilities for each class
    prior_pos = float(class_pos.shape[0]) / n
    prior_neg = float(class_neg.shape[0]) / n

    print("Prior probabilities:")
    print("  P(C+):", prior_pos)
    print("  P(C-):", prior_neg)

    Xtest = np.loadtxt(testingFile)
    nn = Xtest.shape[0]  # Number of test samples

    # Initialize counters for evaluation
    tp = 0  
    tn = 0  
    fp = 0  
    fn = 0  

    for i in range(nn):
        sample = Xtest[i, :d]
        true_label = Xtest[i, d]

        likelihood_pos = 1.0
        likelihood_neg = 1.0
        for j in range(d):
            likelihood_pos *= gaussian_pdf(sample[j], mean_pos[j], std_pos[j])
            likelihood_neg *= gaussian_pdf(sample[j], mean_neg[j], std_neg[j])

        posterior_pos = likelihood_pos * prior_pos
        posterior_neg = likelihood_neg * prior_neg

        predicted = 1 if posterior_pos >= posterior_neg else -1
        #updated counters
        if true_label == 1 and predicted == 1:
            tp += 1
        elif true_label == 1 and predicted == -1:
            fn += 1
        elif true_label == -1 and predicted == 1:
            fp += 1
        elif true_label == -1 and predicted == -1:
            tn += 1

    print("  True Positives:", tp)
    print("  False Positives:", fp)
    print("  True Negatives:", tn)
    print("  False Negatives:", fn)

    accuracy = float(tp + tn) / nn
    precision = float(tp) / (tp + fp) if (tp + fp) > 0 else 0
    recall = float(tp) / (tp + fn) if (tp + fn) > 0 else 0

    print("  Accuracy:", accuracy)
    print("  Precision:", precision)
    print("  Recall:", recall)

if __name__ == "__main__":
    dataset = "iris"
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()

    if dataset == "iris":
        print("Running Naive Bayes classifier on the iris dataset...\n")
        run_naive_bayes("irisTraining.txt", "irisTesting.txt")
    elif dataset == "buy":
        print("Running Naive Bayes classifier on the buy dataset...\n")
        run_naive_bayes("buyTraining.txt", "buyTesting.txt")
    else:
        print("Invalid dataset type provided. Please choose 'iris' or 'buy'.")

#make sure you have numpy library installed by using the command: "pip install numpy"
#run the buy dataset using the command: "python HW_4_skeleton.py buy"
