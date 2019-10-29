import matplotlib.pyplot as plt
from neural_network import build_model,predict,plot_decision_boundary
import numpy as np
from sklearn.datasets import make_moons

np.random.seed(0)
X, y = make_moons(200,noise=0.20)
#plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap=plt.cm.Spectral)
nnhdim = 1
plt.subplot(5 , 2 , 1)
#plt.title( ’HiddenLayerSize %d’ %  nnhdim )
model  =  build_model(X,y ,   nnhdim) 
plot_decision_boundary(lambda x : predict(model,x),X, y )
plt.show()