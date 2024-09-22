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

## Re-grouping

### Drver

#### PREMOVE

1. Going straight; (1)
2. Decelerating & stopping; (2, 5, 7, 9, 13)
3. Accelerating & starting; (3, 4, 8)
4. Turning; (10, 11, 12)
5. Changing lanes; (6, 15, 16, 17)
5. Negotiating a curve; (14)


#### PRELOC

1. Stayed in original travel lane; (1)
2. Stayed on roadway and left original lane; (2, 3)
3. Departed roadway; (4, 5)
4. Entered roadway; (6, 7)

---

### Traffic Env

#### Traffic Condition
0. No traffic control; (0)
1. Traffic control signal (not RR crossing); (1)
2. Stop sign; (2)
3. Yield sign; (3, 4, 5)
4. Warning sign; (6, 7, 8)

#### Related to Intersection
0. Non-interchange area and non-junction; (0)
1. Interchange area related; (1)
2. Intersection related/non-interchange; (2)
3. Driveway/alley access related/non-interchange;(3, 4, 5)

#### Uphill or Downhill
1. Level; (1)
2. Uphill grade (>2%); (2, 3)
3. Downhill grade (>2%); (4, 5)

#### Lighting Condition
1. Daylight; (1)
2. Dark; (2)
3. Dark but lighted; (3)
4. Dawn or Dusk; (4, 5)

#### Surface Type
1. Concrete;(1)
2. Bituminous(asphalt); (2)
3. Other; (3, 4, 5)

#### Surface Condition
1. Dry; (1)
2. Wet; (2)
3. Snow/Slush/Frost; (3, 4, 5, 6)

#### Climate
1. Clear; (1)
2. Rain; (2)
3. Cloudy; (8)
4. Fog, Soil, Sand; (5, 6, 7)
5. Extreme weather (Hail, Snow, Drizzle); (3, 4, 9, 10)