from sklearn.metrics import precision_recall_fscore_support


def solution(true_labels, predicted_labels):
    precision_scores, recall_scores, f1_scores, supports = (
        precision_recall_fscore_support(true_labels, predicted_labels))
    print(precision_scores[0])
