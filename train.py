from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from nyaggle.experiment import run_experiment
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from preprocess import preprocess, category_encode, preprocess_land_price

data_path = Path("resources")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))


def load_dataset():
    train = pd.read_csv(data_path / "train_data.csv")
    test = pd.read_csv(data_path / "test_data.csv")
    rename_pairs = {
        "所在地コード": "市区町村コード", "建蔽率": "建ぺい率（％）",
        "容積率": "容積率（％）", "駅名": "最寄駅：名称",
        "地積": "面積（㎡）", "市区町村名": "市区町村名",
        '前面道路の幅員': '前面道路：幅員（ｍ）', "前面道路の方位区分": "前面道路：方位",
        "前面道路区分": "前面道路：種類", "形状区分": "土地の形状",
        "用途区分": "都市計画", '用途': '地域'
    }
    land_price = pd.read_csv(data_path / "published_land_price.csv",
                             dtype={'利用の現況': str})
    land_price = land_price.rename(columns=rename_pairs)
    return train, test, land_price


def current_status_of_use(land_price):
    riyo_list = np.array(
        ['住宅', '店舗', '事務所', '_', '_',
         '_', '工場', '倉庫', '_', '_', '_', '_',
         '作業場', '_', 'その他', '_', '_']
    )
    riyo_now = [[0] * (17 - len(num)) + list(map(int, list(num)))
                for num in land_price['利用の現況'].values]
    riyo_now = np.array(riyo_now)
    riyo_lists = ['、'.join(riyo_list[onehot.astype('bool')]) for onehot in
                  riyo_now]
    for i in range(len(riyo_lists)):
        if 'その他' in riyo_lists[i]:
            riyo_lists[i] = riyo_lists[i].replace('その他', land_price.loc[
                i, '利用状況表示'])
        riyo_lists[i] = riyo_lists[i].replace('_', 'その他').replace('、雑木林',
                                                                  '').replace(
            '、診療所', '').replace('、車庫', '').replace('、集会場', '') \
            .replace('、寄宿舎', '').replace('、駅舎', '').replace('、劇場', '').replace(
            '、物置', '').replace('、集会場', '').replace('、映画館', '') \
            .replace('、遊技場', '').replace('兼', '、').replace('、建築中',
                                                           'その他').replace(
            '、試写室', '').replace('、寮', '').replace('、保育所', '') \
            .replace('、治療院', '').replace('、診療所', '').replace('、荷捌所',
                                                             '').replace('建築中',
                                                                         'その他').replace(
            '事業所', '事務所').replace('、営業所', '')
    land_price['利用の現況'] = riyo_lists
    return land_price


def clean_land_price(df):
    target_col = "市区町村名"
    # 東京府中 -> 府中
    df[target_col] = df[target_col].replace(r"^東京", "", regex=True)
    return df


def clean_train_test(df):
    target_col = "市区町村名"
    # 西多摩郡日の出 -> 日の出
    df[target_col] = df[target_col].replace(r"^西多摩郡", "", regex=True)
    df[target_col] = df[target_col].map(lambda x: x.rstrip("市区町村"))
    return df


def add_landp(train, test, land_price):
    # 直近5年のみ対象
    target_cols = ["Ｈ２７価格", "Ｈ２８価格", "Ｈ２９価格", "Ｈ３０価格", "Ｈ３１価格"]
    land_price["landp_mean"] = land_price[target_cols].mean(axis=1)
    landp_mean = land_price.groupby("市区町村名")["landp_mean"].mean().reset_index()
    train = train.merge(landp_mean, on='市区町村名')
    test = test.merge(landp_mean, on='市区町村名')
    return train, test


def add_lat_and_long(train, test, land_price):
    lat_and_long = land_price.groupby("市区町村名")[
        "latitude", "longitude"].mean().reset_index()
    train = train.merge(lat_and_long, on='市区町村名')
    test = test.merge(lat_and_long, on='市区町村名')
    return train, test


def main():
    with open("settings/colum_names.yml", "r", encoding="utf-8") as f:
        rename_dict = yaml.load(f, Loader=yaml.Loader)

    train, test, land_price = load_dataset()
    target_col = "y"
    target = train[target_col]
    target = target.map(np.log1p)
    test[target_col] = -1
    _all = pd.concat([train, test], ignore_index=True)

    land_price = preprocess_land_price(land_price)
    _all = _all.rename(columns=rename_dict)
    land_price = land_price.rename(columns=rename_dict)
    _all = preprocess(_all)

    _all["AreaKey"] = _all['Municipality'] + _all['DistrictName']
    _all["AreaKey"] = _all["AreaKey"].str[:5]

    merge_keys = ['MunicipalityCode', 'AreaKey', 'NearestStation']
    land_price_col = "land_price"
    land_price_rates = [f'land_price_rate_{year}_years_per'
                        for year in [3, 5]]
    # count / std / var はfeatureimpが低いため除外
    agg_funcs = ["sum", "min", "max", "mean"]
    for _ in land_price_rates + [land_price_col]:
        for merge_key in merge_keys:
            group_aggs = land_price.groupby(merge_key)[_].agg(
                agg_funcs)
            group_aggs.columns = [f'{merge_key}_{_}_{agg_func}'
                                  for agg_func in group_aggs.columns]
            _all = pd.merge(_all, group_aggs, on=merge_key, how='left')

    # nanになる値を他カラムの値を用いて埋める
    for _ in land_price_rates + [land_price_col]:
        for agg_func in agg_funcs:
            a = f'AreaKey_{_}_{agg_func}'
            b = f'MunicipalityCode_{_}_{agg_func}'
            c = f'NearestStation_{_}_{agg_func}'
            _all.loc[_all[a].isna(), a] = _all.loc[_all[a].isna(), b]
            _all.loc[_all[c].isna(), c] = _all.loc[_all[c].isna(), a]

    drop_cols = ["id", "Prefecture", "Municipality", "年号", "和暦年数", 'FloorPlan']
    one_hot_cols = ['Structure', 'Use', 'Remarks']
    cat_cols = ['Type', 'Region', 'MunicipalityCode', 'DistrictName',
                'NearestStation', 'LandShape', 'Purpose',
                'Direction', 'Classification', 'CityPlanning', 'Renovation',
                'Period', 'AreaKey']
    _all.drop(columns=drop_cols, inplace=True)

    _all = category_encode(_all, cat_cols + one_hot_cols)

    train = _all[_all[target_col] >= 0]
    test = _all[_all[target_col] < 0]

    train.drop(columns=[target_col], inplace=True)
    test.drop(columns=[target_col], inplace=True)
    del _all

    lightgbm_params = {'metric': 'rmse', 'objective': 'regression',
                       'max_depth': 5, 'num_leaves': 30,
                       'learning_rate': 0.007, 'n_estimators': 30000,
                       'min_child_samples': 20, 'subsample': 0.8,
                       'colsample_bytree': 1, 'reg_alpha': 0, 'reg_lambda': 0,
                       'lambda_l1': 6.125651992416009,
                       'lambda_l2': 8.562552194434452e-06,
                       'feature_fraction': 0.484, 'bagging_fraction': 1.0,
                       'bagging_freq': 0}

    fit_params = {
        "early_stopping_rounds": 100,
        "verbose": 5000
    }

    n_splits = 4
    kf = KFold(n_splits=n_splits)

    logging_directory = "resources/logs/lightgbm/{time}"
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging_directory = logging_directory.format(time=now)
    lgb_result = run_experiment(lightgbm_params,
                                X_train=train,
                                y=target,
                                X_test=test,
                                eval_func=rmse,
                                cv=kf,
                                fit_params=fit_params,
                                logging_directory=logging_directory)

    # too long name
    submission = lgb_result.submission_df
    submission[target_col] = submission[target_col].map(np.expm1)
    # replace minus values to 0
    _indexes = submission[submission[target_col] < 0].index
    submission.loc[_indexes, target_col] = 0
    # index 0 to 1
    submission["id"] += 1
    sub_path = Path(logging_directory) / "{}.csv".format(now)
    submission.to_csv(sub_path, index=False)


if __name__ == '__main__':
    main()
