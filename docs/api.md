# API reference

```{contents} List of modules
---
local:
depth: 1
---
```

## `rubella.data_obj` - DataObject

```{eval-rst}
.. currentmodule:: rubella.data_obj

.. autosummary::
   :nosignatures:

   DataObject

.. automodule:: rubella.data_obj
   :members:
   :undoc-members:
```

## `rubella.basic_fc` - Basic functional connectivity

```{eval-rst}
.. currentmodule:: rubella.basic_fc

.. autosummary::
   :nosignatures:

   seed_based_FC
   compute_dual_regression
   compute_global_connectivity
   compute_FC_matrix
   get_sliding_windows

.. automodule:: rubella.basic_fc
   :members:
   :undoc-members:
```

## `rubella.utils` - Data format utilities

```{eval-rst}
.. currentmodule:: rubella.utils

.. autosummary::
   :nosignatures:

   load_img_to_masked_array
   get_img_from_array
   get_concat_timeseries

.. automodule:: rubella.utils
   :members:
   :undoc-members:
```

## `rubella.plotting` - Plotting

```{eval-rst}
.. currentmodule:: rubella.plotting

.. autosummary::
   :nosignatures:

   plot_brain
   plot_many_brains
   gen_slices_array

.. automodule:: rubella.plotting
   :members:
   :undoc-members:
```

## `rubella.qc.report` - Quality control: report

```{eval-rst}
.. currentmodule:: rubella.qc.report

.. autosummary::
   :nosignatures:

   group_stats_QC_plot

.. automodule:: rubella.qc.report
   :members:
   :undoc-members:
```

## `rubella.qc.utils` - Quality control: utilities

```{eval-rst}
.. currentmodule:: rubella.qc.utils

.. autosummary::
   :nosignatures:

   get_group_stats
   get_dice_list
   get_avg
   get_threshold_masks

.. automodule:: rubella.qc.utils
   :members:
   :undoc-members:
```

## `rubella.qpp.main` - Quasi-periodic patterns (QPP)

```{eval-rst}
.. currentmodule:: rubella.qpp.main

.. autosummary::
   :nosignatures:

   compute_QPP
   count_QPP_occurence

.. automodule:: rubella.qpp.main
   :members:
   :undoc-members:
```

## `rubella.caps.main` - Co-activation patterns (CAPs)

```{eval-rst}
.. currentmodule:: rubella.caps.main

.. autosummary::
   :nosignatures:

   find_best_pairs
   reorder_hierarchical
   reorder_Cmodes
   CAPs

.. automodule:: rubella.caps.main
   :members:
   :undoc-members:
```

## `rubella.caps.cluster` - CAPs: clustering

```{eval-rst}
.. currentmodule:: rubella.caps.cluster

.. autosummary::
   :nosignatures:

   pairwise_correlations
   correlation_distances
   cosine_similarity
   cosine_distances
   define_distance
   pre_normalize_distance
   kmeans
   init_centroids
   kmeans_single_rep

.. automodule:: rubella.caps.cluster
   :members:
   :undoc-members:
```

## `rubella.caps.derivatives` - CAPs: derivatives

```{eval-rst}
.. currentmodule:: rubella.caps.derivatives

.. autosummary::
   :nosignatures:

   get_caps_probs
   convert_to_dwell_time
   estimate_transitions
   estimate_transitions_3steps

.. automodule:: rubella.caps.derivatives
   :members:
   :undoc-members:
```

## `rubella.caps.stability` - CAPs: stability

```{eval-rst}
.. currentmodule:: rubella.caps.stability

.. autosummary::
   :nosignatures:

   plot_robustness
   plot_percent_pass
   silhouette
   cluster2data_corr
   plot_cluster2data_corr
   stability_report

.. automodule:: rubella.caps.stability
   :members:
   :undoc-members:
```
