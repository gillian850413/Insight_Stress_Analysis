import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# clf_dict = {'log reg': LogisticRegression(random_state=42),
#             'naive bayes': GaussianNB(),
#             'linear svc': LinearSVC(random_state=42),
#             'ada boost': AdaBoostClassifier(n_estimators=100, random_state=42),
#             'gradient boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
#             'CART': DecisionTreeClassifier(random_state=42),
#             'random forest': RandomForestClassifier(n_estimators=100, random_state=42)}


def classification(clf, X_train, y_train, X_test, y_test):
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # cm = confusion_matrix(y_test, y_pred)
    accuracy =  accuracy_score(y_pred, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    df = pd.DataFrame([('Accuracy', accuracy),
                        ('Precision', precision),
                        ('Recall', recall),
                        ('F1-Score', f1)],
                       columns=('Matrix', 'score'))
    return df


