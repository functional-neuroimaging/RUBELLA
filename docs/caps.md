# CAPs analysis (in progress)
A set of functions (in subfolder `rubella/caps/`) for deriving co-activation patterns (CAPs) using k-means clustering, along with complementary utilitaries for evaluating cluster dimensionality/stability, or computing downstream derivatives (e.g. CAP transition probabilities).

- `rubella/caps/main.py`: contains the main CAPs class that handles deriving clusters, and associated distance metrics.

## Example(s)
```py
# First generate concatenated timeseries across the whole dataset
# data_obj is a RUBELLA DataObject
from rubella.utils import get_concat_timeseries
concat_timeseries = get_concat_timeseries(data_obj.timeseries_dict, data_obj.img_id_list)

from rubella.caps.main import CAPs # importing the main CAP function
X = concat_timeseries
k=8 # the number of clusters to derive
# fitting the CAPs
caps = CAPs(n_clusters=k, distance='cosine', n_rep=5,  max_iter=100, init='k-means++', random_state=1, centroid_order='Cmodes').fit_kmeans(X)
# accessing the generated centroids and labels
centroids=caps.cluster_centers_
labels=caps.labels_
```