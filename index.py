import pandas as pd

dataset = pd.read_csv("novo_dataset.csv")

dataset.head(10)

dataset["classe"].value_counts()

neutroSample = dataset[dataset.classe == "NEUTRO"].sample(500, random_state=1)
positivoSample = dataset[dataset.classe == "POSITIVO"].sample(500, random_state=1)

dataset[dataset.classe == "NEUTRO"] = neutroSample
dataset[dataset.classe == "POSITIVO"] = positivoSample

dataset = dataset.dropna()

len(dataset)

dataset = dataset[dataset.classe != "NEUTRO"]

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

pipeline = Pipeline([
  ("classifier", LogisticRegression(max_iter=1000, random_state=1))
  #("classifier", svm.SVC(random_state=1))
  #("classifier", tree.DecisionTreeClassifier(random_state=1))
  #("classifier", MultinomialNB())
  #("classifier", RandomForestClassifier(random_state=1))
  #("classifier", GradientBoostingClassifier(random_state=1))
])

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics

strat_cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

avg_acc = 0.0
avg_f1 = 0.0
i = 1

X = dataset.drop(columns=["classe"])
y = dataset["classe"]

for train_index, test_index in strat_cv.split(X, y):

  print(f"\n--------------------- AVALIAÇÃO {i} ---------------------\n")
  print()

  clone_pipeline = clone(pipeline)

  X_train = X.iloc[train_index]
  y_train = y.iloc[train_index]
  X_test = X.iloc[test_index]
  y_test = y.iloc[test_index]

  clone_pipeline.fit(X_train, y_train)
  y_pred = clone_pipeline.predict(X_test)

  print(metrics.classification_report(y_test,y_pred,digits=3))
  print(pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Predito'], margins=True))

  avg_acc += accuracy_score(y_test,y_pred)

  prec,rec,f1,supp = score(y_test, y_pred, average='macro')
  avg_f1 += f1

  i += 1

print("\n---> RESULTADOS DA VALIDAÇÃO CRUZADA COM 10 AVALIAÇÕES:")
print(f"Acurácia: {avg_acc/10.0:.4f}")
print(f"F1-score (macro): {avg_f1/10.0:.4f}")

