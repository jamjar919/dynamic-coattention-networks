import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import os
import pickle
from scipy.special import softmax

path = os.path.dirname(os.path.abspath(__file__))+"/svmdata"
print(path)
alphas_file = path+"/alphas_rawv2.pkl"
betas_file = path+"/betas_rawv2.pkl"
labels_file = path+"/labelsv2.pkl"

print(labels_file)

with open(labels_file, "rb") as f:
    labels = pickle.load(f)

with open(alphas_file,"rb") as f :
    alphas = pickle.load(f)

with open(betas_file,"rb") as f :
    betas = pickle.load(f)


labels

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


alphas = softmax(alphas,axis=1)
betas = softmax(betas, axis=1)

max_alphas = [max(alpha) for alpha in alphas]
max_betas = [max(beta) for beta in betas]

sent_alphas = [alpha[0] for alpha in alphas]
sent_betas = [beta[0] for beta in  betas]

# mean_alphas = [np.mean(alpha) for alpha in alphas]
# mean_betas = [np.mean(beta) for beta in betas]

X = np.array(list(zip(sent_alphas,sent_betas)))

# X = np.array(list(zip(mean_alphas,mean_betas)))


# model = SVC(kernel='linear', C=1E10)
# model.fit(X, labels)
# plot_svc_decision_function(model)
# score = model.score(X, labels)
# print("Score: ",score)

xfit = np.linspace(0.0,1.0)

labels = np.array(labels)
labels_no_answer = np.argwhere(labels == -1)
X_no_answer = X[labels_no_answer]
X_no_answer = X_no_answer.reshape(X_no_answer.shape[0],2)
print(labels_no_answer)
print(X_no_answer.shape)

plt.figure()
plt.scatter(X_no_answer[:,0],X_no_answer[:,1],s=5)

labels_answer = np.argwhere(labels == 1)
X_answer = X[labels_answer]
X_answer = X_answer.reshape(X_answer.shape[0],2)
print(X_answer.shape)
plt.figure()
plt.scatter(X_answer[:,0],X_answer[:,1],s=5, c="red")

# plt.scatter(X[:,0], X[:,1], c=labels, s=5, cmap='jet')
plt.show()


