# Lint as: python3
#     Copyright 2020 Google LLC
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
"""Isolation Forest Anomaly Detector."""
from madi.detectors.base_detector import BaseAnomalyDetectionAlgorithm
# from detectors.base_detector import BaseAnomalyDetectionAlgorithm

import madi.utils.sample_utils as sample_utils
# import utils.sample_utils as sample_utils 

import numpy as np
import pandas as pd
import sklearn.ensemble

_CLASS_LABEL = 'class_label'
_NORMAL_CLASS = 1


class NegativeSamplingRandomForestAd(sklearn.ensemble.RandomForestClassifier,
                                     BaseAnomalyDetectionAlgorithm):
  """Anomaly Detector with a Random Forest Classifier and negative sampling."""

  # def __init__(self, *args, sample_ratio=2.0, sample_delta=0.05, **kwargs):
  #   """Constructs a NS-RF Anomaly Detector.

  #   Args:
  #     *args: See the sklearn.ensemble.RandomForestClassifier.
  #     sample_ratio: ratio of negative sample size to positive sample size.
  #     sample_delta: sample extension beypnd min and max limits of pos sample.
  #     **kwargs: See the sklearn.ensemble.RandomForestClassifier.
  #   """
  #   super(NegativeSamplingRandomForestAd, self).__init__(*args, **kwargs)
  #   self._normalization_info = None
  #   self._sample_ratio = sample_ratio
  #   self._sample_delta = sample_delta

  def __init__(self, n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2,
               min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
               max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True,
               oob_score=False, n_jobs=None, random_state=None, verbose=0,
               warm_start=False, class_weight=None):
    """
    Constructs a NS-RF Anomaly Detector.

    Args:
      n_estimators: The number of trees in the forest.
      criterion: The function to measure the quality of a split.
      max_depth: The maximum depth of the trees.
      min_samples_split: The minimum number of samples required to split an internal node.
      min_samples_leaf: The minimum number of samples required to be at a leaf node.
      min_weight_fraction_leaf: The minimum weighted fraction of the sum total of weights required to be at a leaf node.
      max_features: The number of features to consider when looking for the best split.
      max_leaf_nodes: Grow trees with max_leaf_nodes in best-first fashion.
      min_impurity_decrease: A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
      bootstrap: Whether bootstrap samples are used when building trees.
      oob_score: Whether to use out-of-bag samples to estimate the generalization accuracy.
      n_jobs: The number of jobs to run in parallel.
      random_state: Controls the randomness of the estimator.
      verbose: Controls the verbosity when fitting and predicting.
      warm_start: When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble.
      class_weight: Weights associated with classes.
      sample_ratio: Ratio of negative sample size to positive sample size.
      sample_delta: Sample extension beyond min and max limits of positive sample.
    """
    super(NegativeSamplingRandomForestAd, self).__init__(n_estimators=n_estimators,
                                                         criterion=criterion,
                                                         max_depth=max_depth,
                                                         min_samples_split=min_samples_split,
                                                         min_samples_leaf=min_samples_leaf,
                                                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                         max_features=max_features,
                                                         max_leaf_nodes=max_leaf_nodes,
                                                         min_impurity_decrease=min_impurity_decrease,
                                                         bootstrap=bootstrap,
                                                         oob_score=oob_score,
                                                         n_jobs=n_jobs,
                                                         random_state=random_state,
                                                         verbose=verbose,
                                                         warm_start=warm_start,
                                                         class_weight=class_weight,
                                                         )

    self._normalization_info = None
    # self._sample_ratio = sample_ratio
    # self._sample_delta = sample_delta
    print(f'Sample delta assigned')

  def train_model(self, x_train: pd.DataFrame) -> None:
    """Trains a NS-RF Anomaly detector using the positive sample.

    Args:
      x_train: training sample, which does not need to be normalized.
    """
    # TODO(sipple) Consolidate the normalization code into the base class.
    self._normalization_info = sample_utils.get_normalization_info(x_train)
    column_order = sample_utils.get_column_order(self._normalization_info)
    normalized_x_train = sample_utils.normalize(x_train[column_order],
                                                self._normalization_info)

    print(f'Normalized X training set is {normalized_x_train.shape}')
    normalized_training_sample = sample_utils.apply_negative_sample(
        positive_sample=normalized_x_train,
        sample_ratio=2.0,
        sample_delta=0.05) # not using the variable

    print(f'Normalized X training sample is {normalized_training_sample.shape}')
    
    super(NegativeSamplingRandomForestAd, self).fit(
        X=normalized_training_sample[column_order],
        y=normalized_training_sample[_CLASS_LABEL])

  def predict(self, sample_df: pd.DataFrame) -> pd.DataFrame:
    """Performs anomaly detection on a new sample.

    Args:
      sample_df: dataframe with the new datapoints, not normalized.

    Returns:
      original dataframe with a new column labled 'class_prob' rangin from 1.0
      as normal to 0.0 as anomalous.
    """

    sample_df_normalized = sample_utils.normalize(sample_df,
                                                  self._normalization_info)
    column_order = sample_utils.get_column_order(self._normalization_info)
    x = np.float32(np.asarray(sample_df_normalized[column_order]))

    preds = super(NegativeSamplingRandomForestAd, self).predict_proba(x)
    sample_df['class_prob'] = preds[:, _NORMAL_CLASS]
    return sample_df
