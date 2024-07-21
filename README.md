# ConfPre2Post

Code refactoring of old ConfPre2Post repository.

```
Flow:
1. Compare sampling, one-hot and feature selection by different models
   1. Data pre-processing: create different dataset for pretrain;
   2. Pretrain: select best hyper-params of models using Optuna; 
   3. Retrain: based on pretrain hyper-params, retrain model and evaluate performance;
   4. Best Combination Selection: based on retrain results, decide the best combination for CP construction.
2. Conformal Prediction Implementation
   1. Preparation: Naive, APS, CCCP (with/without adaptive weights);
   2. Retrain Conformal Predictor:
      1. Split train data into `(d_train, d_calibration)`;
      2. Train model with `d_train`;
      3. Calibration: find proper threshold `q_hat`;
      4. Evaluation: evaluate CP performance using `Average Set Size` and `Coverage`;
      5. Selection: Choose the best CP method.
3. Analysis:
   1. Check the performance with/without `$\Delta v$`;
   2. Use regular prediction result in 1.4, change it into binary problem and 
compare the performance with other related research;
   3. Analyze hard examples in CP using medical history and other factors to
draw an analytical result.
```

## Point

1. Retrain with rif-data without rus performs worst for only 1% severe data exist.
2. ROS must be tried for both pre-train and retrain.
3. CCCP: different threshold! {0.8, 0.8, 0.9} might be good.


- **Important**: Only divide into with/without one-hot in hyper-params tuning, 
and retrain all data-sets using tha params!
  - Pretrain: ROS data (with/without onehot);
  - Retrain: All sampling sets.