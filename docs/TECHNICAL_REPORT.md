# Technical Report: Customer Segmentation Using Unsupervised Learning

**Learning stage:** Stage 1 --- AI and Machine Learning Foundations\
**Lesson:** 1.6 --- Unsupervised Learning\
**Project:** Customer Segmentation Using Unsupervised Learning\
**Implementation language:** Python

## 1. Executive summary

This project applies unsupervised machine learning techniques to a
synthetic customer dataset containing 600 records. The objective is to
explore patterns in customer behaviour without supplying predefined
target labels.

The implementation covers data generation, feature selection and
scaling, K-Means clustering, cluster-count evaluation, DBSCAN
clustering, Isolation Forest anomaly detection, and Principal Component
Analysis (PCA).

In the observed run:

-   K-Means with three clusters achieved a silhouette score of
    **0.6661**.
-   Three clusters each contained **200** customers.
-   K=3 had the highest silhouette score among the tested K values
    (2--7).
-   DBSCAN discovered **2** clusters and **0** noise points with
    `eps=0.8` and `min_samples=5`.
-   Isolation Forest flagged **30** of 600 records as anomalous.
-   The first two principal components retained **97.25%** of the
    variance.

These results demonstrate the mechanics of several unsupervised learning
methods. Because the data is synthetic and intentionally generated from
three balanced profiles, the findings should not be generalised to real
customer populations.

## 2. Problem definition and objectives

Customer datasets often contain behavioural measurements such as
spending, transaction frequency, and account tenure. Without predefined
labels, it may still be useful to explore whether observations naturally
form groups or whether some records differ substantially from the
overall population.

The project objectives were to:

1.  Generate a reproducible dataset with customer behavioural features.
2.  Prepare numerical features for distance- and variance-based
    algorithms.
3.  discover customer groupings using K-Means and DBSCAN.
4.  Evaluate candidate K-Means cluster counts.
5.  Identify unusual records using Isolation Forest.
6.  Reduce the feature space to two dimensions using PCA.
7.  Save experiment outputs and visualisations for inspection.

This is an unsupervised learning task: no target column containing the
correct customer segment was passed to the algorithms.

## 3. Dataset

### 3.1 Dataset size and features

The generated dataset contains 600 customer records and three
behavioural features, in addition to a customer identifier.

  Column                     Meaning                     Used as a model feature?
  -------------------------- --------------------------- --------------------------
  `customer_id`              Unique record identifier    No
  `monthly_spending`         Customer monthly spending   Yes
  `transactions_per_month`   Monthly transaction count   Yes
  `account_age_months`       Account age in months       Yes

The identifier was excluded because its numeric value does not represent
customer behaviour. Including it could introduce meaningless distance
relationships into the models.

### 3.2 Synthetic profile construction

The generator creates three underlying profiles:

  -------------------------------------------------------------------------
  Synthetic        Monthly spending   Transactions/month Account age centre
  profile                    centre               centre 
  -------------- ------------------ -------------------- ------------------
  Premium                   500,000                   40          48 months

  Regular                   200,000                   20          24 months

  Occasional                 40,000                    5           8 months
  -------------------------------------------------------------------------

Each profile contributes 200 observations. Values are sampled around the
profile centres using normal distributions, then clipped to prevent
negative values.

These profile names describe how the synthetic data was generated. They
are not labels supplied to the clustering models. After clustering, the
model's cluster IDs must be compared with feature summaries before
assigning business interpretations.

## 4. Implementation and project organisation

The project separates responsibilities into Python modules:

  -----------------------------------------------------------------------
  Module                              Responsibility
  ----------------------------------- -----------------------------------
  `generate_data.py`                  Generate and save synthetic
                                      customer records

  `preprocessing.py`                  Load data, select features, and
                                      standardise them

  `clustering.py`                     Fit K-Means and attach cluster
                                      assignments

  `cluster_comparison.py`             Compare K-Means cluster counts and
                                      plot metrics

  `dbscan_experiment.py`              Run DBSCAN and summarise its
                                      assignments

  `anomaly_detection.py`              Run Isolation Forest and identify
                                      flagged records

  `dimensionality_reduction.py`       Run PCA and create a
                                      two-dimensional plot

  `main.py`                           Orchestrate the complete experiment
  -----------------------------------------------------------------------

The workflow is modular so that individual experiments can be modified
without placing all modelling logic in one script.

## 5. Data preprocessing

### 5.1 Feature selection

The three behavioural columns were selected as model inputs.
`customer_id` was excluded.

### 5.2 Standardisation

`StandardScaler` was used to transform each feature so that it has
approximately zero mean and unit variance in the training dataset.

This matters because the features have very different numerical scales.
Monthly spending has values in the hundreds of thousands, while
transaction count and account age are much smaller. Distance-based
methods such as K-Means and DBSCAN can be dominated by large-scale
features if the inputs are not scaled.

The same standardised feature matrix was used for the clustering and PCA
experiments in the observed workflow.

## 6. K-Means clustering

### 6.1 Method

K-Means partitions observations into a specified number of clusters. It
assigns each observation to the nearest centroid and iteratively updates
the centroids to reduce within-cluster squared distances.

The experiment used:

-   `n_clusters=3`
-   `random_state=42`
-   `n_init=10`

The model was fitted to the standardised customer features.

### 6.2 Results

**Silhouette score: 0.6661**

    Cluster ID   Number of records
  ------------ -------------------
             0                 200
             1                 200
             2                 200
     **Total**             **600**

### 6.3 Interpretation

The silhouette score measures how similar an observation is to its own
cluster compared with other clusters. Its range is approximately -1 to
1; a higher value generally indicates better separation and cohesion.

A score of 0.6661 suggests reasonably well-separated clusters for this
dataset.

The equal cluster sizes are consistent with the balanced synthetic
data-generation design. This does not mean that real-world customer
segments will be equally sized.

K-Means cluster IDs are arbitrary. Cluster 0 is not inherently more
important or more premium than cluster 1. A next step is to group the
original customer records by cluster and calculate the mean or median of
each feature to interpret the segments.

## 7. Selecting the number of clusters

### 7.1 Evaluation methods

The experiment compared K values from 2 to 7 using:

-   **Inertia:** the sum of squared distances from observations to their
    assigned cluster centroids. Lower values indicate tighter clusters,
    but inertia generally decreases as K increases.
-   **Silhouette score:** evaluates how well observations fit within
    their assigned clusters compared with neighbouring clusters. Higher
    values generally indicate better-defined clusters.

### 7.2 Results

    K   Inertia   Silhouette score
  --- --------- ------------------
    2    434.17             0.6609
    3    184.03             0.6661
    4    141.81             0.5644
    5    126.44             0.5507
    6    113.09             0.4287
    7    104.71             0.4245

### 7.3 Interpretation

Inertia decreases from 434.17 at K=2 to 104.71 at K=7. This is expected:
adding centroids gives K-Means more flexibility to reduce within-cluster
distances. Therefore, the smallest inertia alone is not a suitable way
to select K.

The silhouette score is highest at K=3 (0.6661), slightly above K=2
(0.6609). Scores decline for K values greater than 3. Considering both
metrics and the known synthetic data construction, K=3 is a reasonable
choice for this experiment.

The generated comparison chart is saved as:

`plots/cluster_comparison.png`

## 8. DBSCAN clustering

### 8.1 Method

DBSCAN is a density-based clustering algorithm. It groups observations
in dense regions and can label observations in sparse regions as noise.
Unlike K-Means, it does not require the number of clusters to be
specified in advance.

The experiment used:

-   `eps=0.8`
-   `min_samples=5`

`eps` defines the neighbourhood radius, while `min_samples` controls the
number of nearby observations required for a point to qualify as a core
point.

### 8.2 Results

  Metric                                             Observed result
  ------------------------------------------------ -----------------
  Clusters discovered                                              2
  Noise points                                                     0
  Cluster 0                                                      200
  Cluster 1                                                      400
  Silhouette score printed by the implementation              0.6609

### 8.3 Interpretation and evaluation caveat

With these parameters, DBSCAN returned two clusters and did not classify
any records as noise. This differs from the three balanced groups
returned by K-Means.

The reported silhouette score should be interpreted cautiously. The
current implementation calculates the score using all returned labels,
including the noise label if one exists. In this run there were no noise
points, so that particular issue does not affect the reported value.

DBSCAN is sensitive to feature scaling and parameter selection. Testing
a range of `eps` and `min_samples` values is necessary before deciding
whether this configuration is appropriate.

## 9. Anomaly detection using Isolation Forest

### 9.1 Method

Isolation Forest is an anomaly-detection algorithm based on the idea
that unusual observations can often be isolated from the rest of the
data using fewer random partitioning steps.

The experiment used:

-   `contamination=0.05`
-   `n_estimators=100`
-   `random_state=42`

The model returned `1` for normal observations and `-1` for observations
flagged as anomalies.

### 9.2 Results

  Classification     Number of records
  ---------------- -------------------
  Normal (`1`)                     570
  Anomaly (`-1`)                    30
  **Total**                    **600**

### 9.3 Interpretation

The model flagged 30 records, which is 5% of the dataset. This is
consistent with the configured contamination setting.

An anomaly is not automatically a fraudulent, incorrect, or harmful
record. It is an observation that the model considers unusual relative
to the dataset. In a real application, flagged records would require
investigation, domain rules, and possibly human review.

The contamination parameter is an assumption used to set the model's
decision threshold; it is not proof that exactly 5% of real customers
are anomalous.

## 10. Dimensionality reduction using PCA

### 10.1 Method

Principal Component Analysis transforms correlated numerical features
into a set of orthogonal components ordered by the amount of variance
they explain.

The experiment reduced the three standardised input features to two
components using:

-   `n_components=2`
-   `random_state=42`

The two-dimensional representation was used for visualisation. K-Means
cluster labels were supplied as colours to help inspect the
distribution.

### 10.2 Results

  Component                 Explained variance
  ----------------------- --------------------
  Principal Component 1                 92.89%
  Principal Component 2                  4.36%
  **Total retained**                **97.25%**

### 10.3 Interpretation

The first two components preserve 97.25% of the variance in the three
original standardised features. Therefore, reducing the data from three
dimensions to two retains most of the variation for this dataset.

PCA is a dimensionality-reduction technique, not a clustering algorithm.
It does not assign customer segments. In this project, the cluster
labels produced by K-Means were used to colour the PCA scatter plot.

The generated visualisation is saved as:

`plots/pca_visualization.png`

## 11. Summary of experimental results

  -----------------------------------------------------------------------
  Technique               Purpose                 Observed result
  ----------------------- ----------------------- -----------------------
  K-Means                 Partition customers     3 clusters; silhouette
                          into a chosen number of 0.6661
                          clusters                

  K-Means evaluation      Compare candidate       Highest tested
                          cluster counts          silhouette at K=3

  DBSCAN                  Discover density-based  2 clusters; 0 noise
                          groups and noise        points

  Isolation Forest        Identify unusual        30 anomalies
                          observations            

  PCA                     Reduce features for     2 components retain
                          visualisation           97.25% variance
  -----------------------------------------------------------------------

## 12. Limitations

1.  **Synthetic data:** The dataset was generated from three
    deliberately separated and balanced profiles. Real customer data may
    be noisy, imbalanced, overlapping, or affected by missing values.
2.  **Known construction:** The underlying synthetic profiles are known
    to the developer, but the models do not receive those profile
    labels. This is a useful learning setup, not a fully independent
    test of real-world segmentation quality.
3.  **Limited feature set:** Only three behavioural features were used.
    Real segmentation may require recency, customer tenure, product
    usage, geography, risk, or other relevant variables.
4.  **Parameter sensitivity:** K-Means and DBSCAN results depend on
    modelling choices, including K, distance geometry, `eps`, and
    `min_samples`.
5.  **Anomaly interpretation:** Isolation Forest flags statistical
    outliers, not confirmed fraud or business-rule violations.
6.  **No external validation:** There is no labelled real-world dataset
    or business outcome used to validate the discovered segments.

## 13. Future improvements

-   Calculate per-cluster mean and median feature values and assign
    evidence-based segment descriptions.
-   Experiment with DBSCAN parameter ranges and inspect cluster/noise
    changes.
-   Evaluate robustness across random seeds and alternative synthetic
    datasets.
-   Add unit tests for data generation, preprocessing, and model
    outputs.
-   Persist fitted models and the fitted scaler for reproducible
    inference.
-   Add a pipeline that prevents preprocessing leakage when training and
    evaluating on separate datasets.
-   Test the workflow on an appropriate real-world dataset.
-   Add business-oriented evaluation, such as whether segments support
    different customer engagement strategies.

## 14. Conclusion

This practical exercise implemented several unsupervised learning
methods in a modular Python project.

K-Means identified three balanced clusters with a silhouette score of
0.6661, and cluster-count evaluation supported K=3 among the tested
values. DBSCAN produced two density-based clusters with no noise points
under the selected configuration. Isolation Forest flagged 30
observations, while PCA reduced the feature space to two components
while retaining 97.25% of the variance.

The exercise demonstrates that different unsupervised algorithms answer
different questions: clustering discovers group structure, anomaly
detection identifies unusual observations, and PCA creates a
lower-dimensional representation. It also highlights the importance of
preprocessing, evaluation, parameter selection, and careful
interpretation.

## 15. Reproduction

From the project directory, create and activate a virtual environment,
install the dependencies, and execute the entry point:

``` bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

The script generates the dataset, runs the experiments, writes result
CSV files, and saves the visualisations. Exact output paths are
determined by the project configuration.

------------------------------------------------------------------------

**Note:** The numerical results in this report reflect the observed run
recorded during development. Re-running the project with changed code,
data-generation settings, dependency versions, or algorithm parameters
may produce different results.
