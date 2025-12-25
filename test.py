import train as tr
import score
def data_vs_data(data1, data2):
    y_test, y_pred =tr.Logistic_Regression_test(data1, data2)
    score.report_classification(y_test, y_pred)
    print('Logistic Regression successfully done')
    y_test, y_pred =tr.Linear_SVC_test(data1, data2)
    score.report_classification(y_test, y_pred)
    print('LinearSVC successfully done')
    print('Data vs Data Done')

def data_test(data):
    y_test, y_pred = tr.Logistic_Regression(data)
    score.report_classification(y_test, y_pred)
    print('Test Logistic Regression successfully done')
    y_test, y_pred =tr.Linear_SVC(data)
    score.report_classification(y_test, y_pred)
    print('Test Linear SVC successfully done')
    y_test, y_pred =tr.Logistic_Regression_proba(data)
    score.report_prob(y_test, y_pred)
    print('Test Logistic Regression Proba successfully done')
    print('Data Test Done')
