# About Conformal Prediction Code

## 1. Three methods are provided as:
### 1.1 Naive Conformal Prediction;
### 1.2 Adaptive Prediction Set (APS);
### 1.3 Class-Conditional Conformal Prediction (CCCP);
Details of each method will be added when combining total program.

## 2. How to use this simple CP code:

To make a conformal prediction after training(which we call split conformal prediction), we need to do: 
1. calibration; 
2. validation.

Both are implemented in the code. 

**For calibration**, what we do is to find a quantile $\hat q$ and use it as 
threshold to construct the conformal set. To find the quantile, you need to use the following command:

```q_hat = calc_qhat_Method(y_softmax, y_cal, alpha=0.1)```

Where: 
1. `y_softmax` is the softmax output of model by calibration data `x_cal`, usually is the output of `model.predict_proba(x)`;
2. `y_cal` is the real label for calibration data;
3. `alpha` is the user specified coverage, which roughly guarantee a coverage of `1-alpha` and is distribution-free.
 Defaultly, it is set as `alpha=0.1` which guarantee a 90% coverage.

**For validation**, first you will need to make conformal prediction by following command:

```conf_sets = get_confsets_Method(softmax_out, q_hat)```

Where:
1. `softmax_out` is the softmax output of a classification model;
2. `q_hat` is the quantile number(s for only CCCP) **obtained by function in calibration**.

After get the conformal set `conf_set`, we could use functions in `eval_CP.py` to evaluate Average Set Size, Coverage 
and draw figures.

For calculate average set size and coverage, we use `ass = calc_average_set_size(conf_set)` 
and `cover = calc_coverage(conf_set, y_true)` to achieve it. For draw correponding figure, `draw_set_sizes(conf_set, cp_type)` and 
`draw_coverage(conf_set, y_true, cp_type)`.

In addition, a metric called _**Size-stratified coverage metric(SSC)**_, which evaluates these metrics
concurrently. The definition is shown below: 

$$SSC Metric = \frac{1}{|\mathcal{L}_g|} \sum_{i \in \mathcal{L}_g} \mathbf{1}\{Y_i^{(val)} \in \mathbf{C}(X_i^{(val)})\}$$

In this repository, you can use  `calc_ssc_metric(conf_set, y_true)` to calculate the **_SSC Metric_**.