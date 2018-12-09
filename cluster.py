import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#plt.figure(figsize=(12,12))
np.random.seed(5)

data = pd.read_csv("./data.csv")

data.key = np.where(data.key > 10.0, 10.0, data.key)

print(data.head())
print(data.describe())
print(data.isna().sum())

# drop unneeded columns for now
X = data.drop(['artist_name', 'artist_mbtags'], axis=1)

# encode artist id as integers
labelEncoder = LabelEncoder()
labelEncoder.fit(X["artist_id"])
X["artist_id"] = labelEncoder.transform(X["artist_id"])
print(X.info())

# scale the values
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=63, random_state=0, max_iter=600 )
y = kmeans.fit_predict(X_scaled)
#kmeans.fit(X)
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(y)

# write values back
data['class'] = y
data.to_csv('with_class.csv')

#Ydf = pd.DataFrame(y)
#print(Ydf.describe())


"""countsY = [0] * 63
for val in y:
    countsY[val]+=1
print(countsY)"""

"""fig = plt.figure(0, figsize=(4, 3))
ax = Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=1134)
ax.scatter(X[:,3], X[:,0], X[:,2], c=y)
fig.show()"""