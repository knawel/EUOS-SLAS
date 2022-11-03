from datetime import datetime

# classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier

from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier
# from catboost import CatBoostClassifier

config_data = {
    'dataset_filepath': "../../data/preprocessed"
}

# Tag for name of the model
tag = datetime.now().strftime("_%Y-%m-%d_%H-%M")

config_runtime = {
    'run_name': 'v1' + tag
}

classifiers = {
    # "Logit" : LogisticRegression(),
    "SGD": SGDClassifier(loss='squared_error'),
    "MLP_1": MLPClassifier(alpha=1, solver='sgd', max_iter=5000, learning_rate='adaptive'),
    "K-neigh": KNeighborsClassifier(3),
    "SVN_1": SVC(kernel="linear", max_iter=1000, class_weight='balanced'),
    "Tree": DecisionTreeClassifier(max_depth=8),
    "Tree w": DecisionTreeClassifier(max_depth=8, class_weight='balanced'),
    "Forest 1": RandomForestClassifier(class_weight='balanced'),
    "Forest 2": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Ada 1": AdaBoostClassifier(random_state=0),
    "GBCls": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, random_state=0),
    "LGBMC": LGBMClassifier(class_weight={0: 5, 1: 5, 2: 1}),
    "GaussNB": GaussianNB()

}
