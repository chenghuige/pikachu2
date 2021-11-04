import sys
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':

    test_score_file = sys.argv[1]
    validate_score_file = sys.argv[2]

    labels, scores = [], []
    with open(test_score_file) as file:
        for line in file:
            line_tuple = line.strip().split(",")
            label = int(line_tuple[0])
            score = float(line_tuple[-1])
            labels.append(label)
            scores.append(score)
    test_score = roc_auc_score(labels, scores)

    labels, scores = [], []
    with open(validate_score_file) as file:
        for line in file.readlines()[1:]:
            line_tuple = line.strip().split(",")
            label = float(line_tuple[2])
            score = float(line_tuple[-1])
            labels.append(label)
            scores.append(score)
    validate_score = roc_auc_score(labels, scores)
    if abs(test_score - validate_score) <= 0.001:
        print("ok")
    else:
        print("%s %s" % (test_score, validate_score))
