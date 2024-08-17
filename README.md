# ConfPre2Post

Code refactoring of old ConfPre2Post repository.

## Flow

```
Flow:
1. Compare sampling, one-hot and feature selection by different models
   1. Data pre-processing: create ROS with/without one_hot for pretrain;
   2. Pretrain: select best hyper-params of models using Optuna; 
   3. Retrain: based on pretrain hyper-params, retrain model by different sampling and evaluate performance;
      - Train: Calib: Test = 8: 9: 1 (Calib = Train + 1/2 Test)
   4. Best Combination Selection: based on retrain results, decide the best combination for CP construction.
2. Conformal Prediction Implementation
   1. Preparation: Naive, APS, CCCP (with/without adaptive weights);
   2. Construct Conformal Predictor:
      1. Obtain d_calibration as 1/2 d_test;
      2. Use `(d_train, d_calibration)` as calib_set;
      3. Calibration: find proper threshold `q_hat`;
      4. Evaluation: evaluate CP performance using `Average Set Size` and `Coverage`;
      5. Selection: Choose the best CP method.
3. Analysis:
   1. Check the performance with/without `$\Delta v$` and `maneuver` (not pre-crash but important!);
   2. Use regular prediction result in 1.4, change it into binary problem and 
compare the performance with other related research;
   3. Analyze hard examples in CP using medical history and other factors to
draw an analytical result.
```

## Point

1. Retrain with rif-data without rus performs worst for only 1% severe data exist.

2. SMOTE-based techs not proper for this task!
  - SMOTE-based techs interpolates between minority class samples to generate new synthetic samples,
however, quality of generated data seems to be not that good for non-severe and severe data are like each other.

---

## Analysis 

### Data relationship Analysis

1. Analyze relationship between injury severity and medical records (no need to divide by crash typ);
- Caution: This feature will be blank for **uninjured occupants**, which means it could only be used for
   analyzing the effect of Slight and Injury Severity!!!
  
- Conclusion:
  - **Clear positive correlation** was observed between past medical records and  injury severity;

2. Divide cases by Crash Type, and analyze relationship between injury risk and maneuver;
- Discussion: No obvious correlation was observed!
    
3. Check hard cases based on above information (including $\Delta V$).
- Discussion: In fact, medical records and $\Delta V$ provide nearly no extra info for enhancing recalling severe.