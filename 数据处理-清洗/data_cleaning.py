from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    _csv =  Path("data/chicago_taxi_train.csv")
    if not _csv.exists():
        _csv = Path("../data/chicago_taxi_train.csv")
    chicago_taxi_dataset = pd.read_csv(_csv)

    # chicago_taxi_dataset.info()
    # 查看数据情况
    # <class 'pandas.core.frame.DataFrame'>
    # RangeIndex: 31694 entries, 0 to 31693
    # Data columns (total 18 columns):
    # #   Column                  Non-Null Count  Dtype  
    # ---  ------                  --------------  -----  
    # 0   TRIP_START_TIMESTAMP    31694 non-null  object 
    # 1   TRIP_END_TIMESTAMP      31694 non-null  object 
    # 2   TRIP_START_HOUR         31694 non-null  float64
    # 3   TRIP_SECONDS            31694 non-null  int64  
    # 4   TRIP_MILES              31694 non-null  float64
    # 5   TRIP_SPEED              31694 non-null  float64
    # 6   PICKUP_CENSUS_TRACT     13259 non-null  float64
    # 7   DROPOFF_CENSUS_TRACT    14023 non-null  float64
    # 8   PICKUP_COMMUNITY_AREA   28477 non-null  float64
    # 9   DROPOFF_COMMUNITY_AREA  28199 non-null  float64
    # 10  FARE                    31694 non-null  float64
    # 11  TIPS                    31694 non-null  float64
    # 12  TIP_RATE                31694 non-null  float64
    # 13  TOLLS                   31694 non-null  float64
    # 14  EXTRAS                  31694 non-null  float64
    # 15  TRIP_TOTAL              31694 non-null  float64
    # 16  PAYMENT_TYPE            31694 non-null  object 
    # 17  COMPANY                 31694 non-null  object 
    # dtypes: float64(13), int64(1), object(4)
    # memory usage: 4.4+ MB

    # print(chicago_taxi_dataset.describe())
    # TRIP_START_HOUR  TRIP_SECONDS    TRIP_MILES    TRIP_SPEED  PICKUP_CENSUS_TRACT  DROPOFF_CENSUS_TRACT  ...          FARE          TIPS      TIP_RATE         TOLLS        EXTRAS    TRIP_TOTAL
    # count     31694.000000  31694.000000  31694.000000  31694.000000         1.325900e+04          1.402300e+04  ...  31694.000000  31694.000000  31694.000000  31694.000000  31694.000000  31694.000000
    # mean         14.185635   1319.796397      8.289463     20.256544         1.703153e+10          1.703140e+10  ...     23.905210      3.246130     12.965785      0.002744      2.410468     29.755053
    # std           5.159454    928.932873      7.265672     11.110390         3.827747e+05          3.408106e+05  ...     16.970022      4.284567     15.517765      0.164343      5.805087     23.145537
    # min           0.000000     60.000000      0.500000      4.000000         1.703102e+10          1.703101e+10  ...      3.250000      0.000000      0.000000      0.000000      0.000000      3.250000
    # 25%          10.500000    548.000000      1.720000     11.000000         1.703108e+10          1.703108e+10  ...      9.000000      0.000000      0.000000      0.000000      0.000000     10.750000
    # 50%          14.750000   1081.000000      5.920000     17.600000         1.703132e+10          1.703132e+10  ...     18.750000      2.000000     12.200000      0.000000      0.000000     21.700000
    # 75%          18.000000   1888.000000     14.500000     27.500000         1.703198e+10          1.703184e+10  ...     38.750000      5.000000     20.800000      0.000000      4.000000     48.265000
    # max          23.750000   7140.000000     68.120000     64.800000         1.703198e+10          1.703198e+10  ...    159.250000     60.000000    648.600000     27.000000     85.000000    248.500000

    # [8 rows x 14 columns]
    
    # print(chicago_taxi_dataset.isna())
    #                 TRIP_START_TIMESTAMP  TRIP_END_TIMESTAMP  TRIP_START_HOUR  TRIP_SECONDS  TRIP_MILES  TRIP_SPEED  PICKUP_CENSUS_TRACT  ...   TIPS  TIP_RATE  TOLLS  EXTRAS  TRIP_TOTAL  PAYMENT_TYPE  COMPANY
    # 0                     False               False            False         False       False       False                 True  ...  False     False  False   False       False         False    False
    # 1                     False               False            False         False       False       False                 True  ...  False     False  False   False       False         False    False
    # 2                     False               False            False         False       False       False                False  ...  False     False  False   False       False         False    False
    # 3                     False               False            False         False       False       False                False  ...  False     False  False   False       False         False    False
    # 4                     False               False            False         False       False       False                False  ...  False     False  False   False       False         False    False
    # ...                     ...                 ...              ...           ...         ...         ...                  ...  ...    ...       ...    ...     ...         ...           ...      ...
    # 31689                 False               False            False         False       False       False                 True  ...  False     False  False   False       False         False    False
    # 31690                 False               False            False         False       False       False                 True  ...  False     False  False   False       False         False    False
    # 31691                 False               False            False         False       False       False                 True  ...  False     False  False   False       False         False    False
    # 31692                 False               False            False         False       False       False                 True  ...  False     False  False   False       False         False    False
    # 31693                 False               False            False         False       False       False                False  ...  False     False  False   False       False         False    False

    # [31694 rows x 18 columns]

    # print(chicago_taxi_dataset.isna().sum())
    # TRIP_SPEED                    0
    # PICKUP_CENSUS_TRACT       18435
    # DROPOFF_CENSUS_TRACT      17671
    # PICKUP_COMMUNITY_AREA      3217
    # DROPOFF_COMMUNITY_AREA     3495
    # FARE                          0
    # TIPS                          0
    # TIP_RATE                      0
    # TOLLS                         0
    # EXTRAS                        0
    # TRIP_TOTAL                    0
    # PAYMENT_TYPE                  0
    # COMPANY                       0
    # dtype: int64

    # print(chicago_taxi_dataset.duplicated().sum())
    # >> 0

    # print(chicago_taxi_dataset.quantile(numeric_only=True, q=0.25))
    # TRIP_SECONDS              5.480000e+02
    # TRIP_MILES                1.720000e+00
    # TRIP_SPEED                1.100000e+01
    # PICKUP_CENSUS_TRACT       1.703108e+10
    # DROPOFF_CENSUS_TRACT      1.703108e+10
    # PICKUP_COMMUNITY_AREA     8.000000e+00
    # DROPOFF_COMMUNITY_AREA    8.000000e+00
    # FARE                      9.000000e+00
    # TIPS                      0.000000e+00
    # TIP_RATE                  0.000000e+00
    # TOLLS                     0.000000e+00
    # EXTRAS                    0.000000e+00
    # TRIP_TOTAL                1.075000e+01
    # Name: 0.25, dtype: float64

    # chicago_taxi_dataset = chicago_taxi_dataset.dropna(how="any")

    # 将所有 NaN 元素替换为 0。
    # chicago_taxi_dataset = chicago_taxi_dataset.fillna(0)

    # 将列“PICKUP_CENSUS_TRACT”、“DROPOFF_CENSUS_TRACT”中的所有 NaN 元素分别替换为 0、1。
    # values = {"PICKUP_CENSUS_TRACT": 0, "DROPOFF_CENSUS_TRACT": 1}
    # chicago_taxi_dataset = chicago_taxi_dataset.fillna(value=values)

    chicago_taxi_dataset = chicago_taxi_dataset.drop_duplicates()