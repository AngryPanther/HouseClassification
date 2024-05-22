from sklearn.metrics import classification_report

def solution(true_labels, predicted_labels):
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=['ApPlE', 'BaNaNa', 'OrAnGe', 'PeAr'],
        )
    )