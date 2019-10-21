# Generate a dataset and plot it
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from neural_network import build_model, predict, plot_decision_boundary

np.random.seed(0)
X, y = make_moons(200, noise=0.20)

plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral) 
plt.figure(figsize=(16, 32))

hidden_layer_dimensions = [1,2,3,4]

for i, nn_hdim in enumerate(hidden_layer_dimensions):
    plt.subplot(5, 2, i+1)
    plt.title('HiddenLayerSize%d' % nn_hdim)
    model = build_model(X, y, nn_hdim)
    plot_decision_boundary(lambda x: predict(model, x), X, y)
plt.show() 