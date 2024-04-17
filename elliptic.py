# elliptic envelope for imbalanced classification
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.base import OutlierMixin
from sklearn.covariance import MinCovDet
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import warnings  
  
# warnings.filterwarnings('ignore')  

class EllipticEnvelopeClassifier:


    def __init__(self,contamination=0.1):
        self.mcd = MinCovDet(support_fraction=1)
        self.contamination = contamination

    def fit(self, X):
        self.mcd.fit(X)
        self.offset_ = np.percentile(-self.mcd.dist_, 100.0 * self.contamination)
        return self
    
    def mahalanobis(self, X):
        dist = pairwise_distances(X, self.mcd.location_[np.newaxis, :], metric="mahalanobis", VI=self.mcd.get_precision())
        return np.reshape(dist, (len(X),)) ** 2

    def predict(self, X):
        negative_mahal_dist = -self.mahalanobis(X)
        values = negative_mahal_dist - self.offset_
        outlier_val = np.full(values.shape[0], 1, dtype=int)
        outlier_val[values >= 0] = 0

        return outlier_val



df = pd.read_csv('http.csv')
df = df.drop_duplicates()
X = df.drop(columns=['attack']).values
Y = df['attack'].values
trainX, testX, trainy, testy = train_test_split(X, Y, test_size=0.2, random_state=42)
# define outlier detection model
# model = EllipticEnvelopeClassifier(contamination=0.05)
model = EllipticEnvelopeClassifier(contamination=1e-4)
# fit on majority class
trainX = trainX[trainy==0]
model.fit(trainX)
# detect outliers in the test set
testy_pred = model.predict(testX)
cm = confusion_matrix(testy>0, testy_pred>0)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Inlier','Outlier'])
disp.plot()
print("Outlier predictions:",np.sum(testy_pred>0))
print("Inlier predictions:",np.sum(testy_pred<=0))
print("Outlier actuals:",np.sum(testy>0))
print("Inlier actuals:",np.sum(testy<=0))
print(f'Accuracy score: {accuracy_score(testy>0, testy_pred>0) :>.3%}')
print(f'F1 score: {f1_score(testy>0, testy_pred>0) :>.3}')
plt.show()
