from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def report_classification(y_test, y_pred):

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def report_prob(y_test, y_proba, threshold = 0.5):

    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    y_pred_tuned = (y_proba >= threshold).astype(int)

    print(confusion_matrix(y_test, y_pred_tuned))
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.xlabel("Threshold")
    plt.legend()
    plt.title("Precision-Recall vs Threshold")
    plt.savefig("Output/SVC_vs_Regression/precision-recall_threshold.png", dpi=300, bbox_inches='tight')
    plt.show()