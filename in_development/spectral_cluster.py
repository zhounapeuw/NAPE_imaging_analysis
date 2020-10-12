from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, silhouette_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, KMeans
import scipy.stats as stats
from sklearn.metrics import roc_auc_score as auROC
import statsmodels.api as sm



#############

populationdata_for_key = {}
for k, key in enumerate(keys_withtrials):
    # print key
    print align_baselinesubtracted[key][trials_of_interest_dict[key], :, :].shape
    print numframesforcue
    populationdata = calculate_centraltendency_for_rois(
        align_baselinesubtracted[key][trials_of_interest_dict[key], :, :],
        numframesforcue - 1, centraltendency=centraltendency)
    populationdata_for_key[key] = populationdata
    cmin_for_key[k] = np.amin(populationdata)
    cmax_for_key[k] = np.amax(populationdata)
    if ('plus' in key) and (temp == 0):
        temp = 1
        # Sort by cue response
        if sortby == 'cue response':
            tempresponse = np.mean(populationdata[:, pre_window_size:pre_window_size + numframesforcue], axis=1)
            # temp=np.divide(1, (np.arange(1,post_window_size)+0.0)**0.5)
            # tempresponse = np.sum(populationdata[:,pre_window_size+1:]*np.tile(np.expand_dims(temp,axis=0), (populationdata.shape[0],1)),
            #                     axis=1)

        # Sort by reward response
        elif sortby == 'reward response':
            tempresponse = np.mean(
                populationdata[:, pre_window_size + numframesforcue:pre_window_size + 2 * numframesforcue],
                axis=1)
        elif sortby == '':
            tempresponse = np.arange(0, numrois)
        sortresponse = np.argsort(tempresponse)[::-1]


########

data_for_pca = np.concatenate([populationdata_for_key[key] for key in sorted(populationdata_for_key)], axis=1)
print data_for_pca.shape

min_variance_explained = 0.9
pca = PCA(n_components=data_for_pca.shape[1], whiten=True)
pca.fit(data_for_pca)
compressed_data = pca.transform(data_for_pca) #transform back to shape n_components x n_timepoints
pca_vectors = pca.components_
print pca_vectors.shape

x = 100*pca.explained_variance_ratio_
xprime = x - (x[0] + (x[-1]-x[0])/(x.size-1)*np.arange(x.size))
threshold = np.argmin(xprime)
# Number of PCs to be kept is defined as the number at which the
# scree plot bends. This is done by simply bending the scree plot
# around the line joining (1, variance explained by first PC) and
# (num of PCs, variance explained by the last PC) and finding the
# number of components just below the minimum of this rotated plot
print 'Number of PCs to keep = %d'%(threshold)