import numpy as np
import pandas as pd
from mojimoji import zen_to_han
from sklearn.preprocessing import LabelEncoder


def category_encode(df, target_cols):
    for col in target_cols:
        df[col] = df[col].fillna("NaN")
        le = LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
    return df


def preprocess(df):
    df = cleaning(df)
    df = built_year(df)
    df = walk_time(df)
    df = area1(df)
    df = area2(df)
    df = maguchi(df)
    df = ldk(df)
    df = num_of_rooms(df)
    df = nearest_station(df)
    # df = period(df)
    # df = region(df)
    df = total_floor_area_div_area(df)
    df = total_floor_area_per_floor(df)
    df = area_div_frontage(df)
    df = frontage_div_breadth(df)
    # df = remarks(df)
    # df = landshape(df)
    # df = structure(df)
    # df = use(df)
    return df


def cleaning(df):
    # zenkaku to hankaku
    df['間取り'] = df['間取り'].map(zen_to_han)
    return df


def preprocess_land_price(land_price):
    land_price['最寄駅：距離（分）'] = land_price['駅距離'] // 50
    land_price.loc[:, '最寄駅：距離（分）'][land_price['最寄駅：距離（分）'] > 120] = 120
    land_price['間口（比率）'] = land_price['間口（比率）'].clip(10, 100)
    land_price['奥行（比率）'] = land_price['奥行（比率）'].clip(10, 100)
    land_price['間口'] = np.sqrt(
        land_price['面積（㎡）'] / land_price['間口（比率）'] / land_price[
            '奥行（比率）']) * land_price['間口（比率）']

    # 東京府中 -> 府中
    land_price["市区町村名"] = land_price["市区町村名"].replace(r"^東京", "",
                                                      regex=True)
    # train/testと統一
    land_price['最寄駅：名称'] = land_price['最寄駅：名称'].str.replace('ケ', 'ヶ')
    land_price["市区町村名"] = land_price["市区町村名"].str.replace('ケ', 'ヶ')
    land_price["面積（㎡）"] = land_price["面積（㎡）"].clip(0, 3000)
    # preprocess 利用の現況
    # 最新の公示価格を対象
    target_col = "Ｈ３１価格"
    # 3 / 5 / 10 / 20 最新年との比
    land_price_rate_3_years_per = land_price[target_col] / \
                                  land_price['Ｈ２８価格']
    land_price_rate_3_years_per = land_price_rate_3_years_per.rename(
        "land_price_rate_3_years_per")
    land_price_rate_5_years_per = land_price[target_col] / \
                                  land_price['Ｈ２６価格']
    land_price_rate_5_years_per = land_price_rate_5_years_per.rename(
        "land_price_rate_5_years_per")
    # land_price_rate_10_years_per = land_price[target_col] / \
    #                                land_price['Ｈ２１価格']
    # land_price_rate_10_years_per = land_price_rate_10_years_per.rename(
    #     "land_price_rate_10_years_per")
    # land_price_rate_20_years_per = land_price[target_col] / \
    #                                land_price['Ｈ１１価格']
    # land_price_rate_20_years_per = land_price_rate_20_years_per.rename(
    #     "land_price_rate_20_years_per")

    target = land_price[target_col]
    target = target.rename("land_price")
    # train/test 取引金額(1,000,000円)表記に合わせる
    target = target / 100000
    # drop_pat_1 = '(Ｓ|Ｈ).+価格'
    # drop_pat_2 = '属性移動(Ｓ|Ｈ).+'
    # 容積率（％）までのカラムを用いる
    land_price = land_price.iloc[:, :41]
    land_price["取引時点"] = 2019
    land_price = land_price.join(target)
    land_price = land_price.join(land_price_rate_3_years_per)
    land_price = land_price.join(land_price_rate_5_years_per)
    # land_price = land_price.join(land_price_rate_10_years_per)
    # land_price = land_price.join(land_price_rate_20_years_per)

    # land_price = land_price.rename({"緯度": "latitude", "経度": "longitude"})
    rep = {'1低専': '第１種低層住居専用地域',
           '2低専': '第２種低層住居専用地域',
           '1中専': '第１種中高層住居専用地域',
           '2中専': '第２種中高層住居専用地域',
           '1住居': '第１種住居地域',
           '2住居': '第２種住居地域',
           '準住居': '準住居地域', '商業': '商業地域', '近商': '近隣商業地域',
           '工業': '工業地域', '工専': '工業専用地域', '準工': '準工業地域', '田園住': '田園住居地域'}
    for key, value in rep.items():
        land_price.loc[:, '都市計画'] = land_price.loc[:, '都市計画'].str.replace(key,
                                                                          value)
    land_price = land_price.rename(columns={'利用の現況': '用途'})

    # 住所番地手前で切り出し
    se = land_price['住居表示'].str.strip('東京都').str.replace('大字',
                                                         '').str.replace(
        '字', '')
    # 番地総当たり
    for num in ['１', '２', '３', '４', '５', '６', '７', '８', '９']:
        se = se.str.split(num).str[0].str.strip()
    land_price['AreaKey'] = se
    land_price['AreaKey'] = land_price['AreaKey'].str[:5]
    return land_price


def built_year(df):
    df['BuildingYear'] = df['BuildingYear'].dropna()
    df['BuildingYear'] = df['BuildingYear'].str.replace('戦前', '昭和20年')
    df['年号'] = df['BuildingYear'].str[:2]
    df['和暦年数'] = df['BuildingYear'].str[2:].str.strip('年').fillna(0).astype(
        int)
    df.loc[df['年号'] == '昭和', 'BuildingYear'] = df['和暦年数'] + 1925
    df.loc[df['年号'] == '平成', 'BuildingYear'] = df['和暦年数'] + 1988
    df['BuildingYear'] = pd.to_numeric(df['BuildingYear'])
    return df


def period(df):
    replace_dict = {'年第': '.',
                    '四半期': '',
                    '１': '0',
                    '２': '25',
                    '３': '50',
                    '４': '75'}
    df["Period"] = pd.to_numeric(df["Period"].replace(replace_dict),
                                 errors='raise')
    return df


def get_num_of_rooms(floor_plan):
    try:
        _num_of_rooms = int(floor_plan[0])
    except ValueError:
        # nan is other value
        if floor_plan == '<NA>' or floor_plan == 'nan':
            return 0
        else:
            return 1
    return _num_of_rooms


def ldk(df):
    df['L'] = df['FloorPlan'].map(lambda x: 1 if 'L' in str(x) else 0)
    df['D'] = df['FloorPlan'].map(lambda x: 1 if 'D' in str(x) else 0)
    df['K'] = df['FloorPlan'].map(lambda x: 1 if 'K' in str(x) else 0)
    df['S'] = df['FloorPlan'].map(lambda x: 1 if 'S' in str(x) else 0)
    df['R'] = df['FloorPlan'].map(lambda x: 1 if 'R' in str(x) else 0)
    df['Maisonette'] = df['FloorPlan'].map(
        lambda x: 1 if 'メゾネット' in str(x) else 0)
    df['OpenFloor'] = df['FloorPlan'].map(
        lambda x: 1 if 'オープンフロア' in str(x) else 0)
    df['Studio'] = df['FloorPlan'].map(lambda x: 1 if 'スタジオ' in str(x) else 0)

    return df


def num_of_rooms(df):
    df['num_of_rooms'] = df['FloorPlan'].map(
        lambda x: get_num_of_rooms(str(x)))
    return df


def walk_time(df):
    df['TimeToNearestStation'] = df['TimeToNearestStation'].replace('30分?60分',
                                                                    '45')
    df['TimeToNearestStation'] = df['TimeToNearestStation'].replace('1H?1H30',
                                                                    '75')
    df['TimeToNearestStation'] = df['TimeToNearestStation'].replace('1H30?2H',
                                                                    '105')
    df['TimeToNearestStation'] = df['TimeToNearestStation'].replace('2H?',
                                                                    '120')
    df['TimeToNearestStation'] = pd.to_numeric(df['TimeToNearestStation'],
                                               errors='coerce')
    return df


def area1(df):
    replace_dict = {'10m^2未満': 9, '2000㎡以上': 2000}
    df['TotalFloorArea'] = pd.to_numeric(
        df['TotalFloorArea'].replace(replace_dict))
    return df


def area2(df):
    replace_dict = {'2000㎡以上': 2000, '5000㎡以上': 5000}
    df['Area'] = pd.to_numeric(df['Area'].replace(replace_dict))

    df.loc[(df['Type'] == '林地') | (df['Type'] == '農地'), 'Area'] *= 0.1
    return df


def region(df):
    replace_dict = {'住宅地': '0',
                    '宅地見込地': '003',
                    '商業地': '005',
                    '工業地': '009'}
    df['Region'] = pd.to_numeric(df['Region'].replace(replace_dict))
    return df


def total_floor_area_div_area(df):
    df['total_floor_area_div_area'] = df['TotalFloorArea'] / df['Area']
    return df


def total_floor_area_per_floor(df):
    df['total_floor_area'] = df['Area'] / df['num_of_rooms']
    return df


def area_div_frontage(df):
    df['area_div_frontage'] = df['Area'] / df['Frontage']
    return df


def frontage_div_breadth(df):
    df['frontage_div_breadth'] = df['Frontage'] / df['Breadth']
    return df


def maguchi(df):
    df['Frontage'] = pd.to_numeric(df['Frontage'].replace('50.0m以上', 50.0))
    return df


def landshape(df):
    # ほぼ長方形 -> 長方形
    df["LandShape"] = df["LandShape"].replace("^ほぼ", "", regex=True)
    return df


def series_split_colum(df, col_name):
    split_df = df[col_name].str.get_dummies(sep="、")
    col_names = ["{}_{}".format(col_name, idx) for idx in
                 range(split_df.shape[1])]
    split_df.columns = col_names
    return df.join(split_df)


def structure(df):
    col_name = "Structure"
    """
    建物の構造: Structure

        ブロック造  木造  軽量鉄骨造  鉄骨造  ＲＣ  ＳＲＣ
    0           0   0      0    0   0    1
    1           0   0      0    0   1    0
    """
    return series_split_colum(df, col_name)


def remarks(df):
    col_name = "Remarks"
    """
    取引の事情等: Remarks

        その他事情有り  他の権利・負担付き  古屋付き・取壊し前提  ...
    0             0          0           0  ...
    1             0          0           0  ...
    """
    return series_split_colum(df, col_name)


def use(df):
    col_name = "Use"
    """
    用途: Use

        その他  事務所  住宅  作業場  倉庫  共同住宅  工場  店舗  駐車場
    0         0    0   0    0   0     0   0   0    0
    1         0    0   0    0   0     0   0   0    0
    2         0    0   1    0   0     0   0   0    0
    3         0    0   1    0   0     0   0   0    0
    4         0    1   1    0   0     0   0   1    0
    """
    return series_split_colum(df, col_name)


def nearest_station(df):
    df['NearestStation'] = df['NearestStation'].fillna('なし')
    df['NearestStation'] = df['NearestStation'].str.replace('(東京)',
                                                            '').str.replace(
        '(神奈川)', '').str.replace('ケ', 'ヶ')
    df['NearestStation'] = df['NearestStation'].str.replace('(メトロ)',
                                                            '').str.replace(
        '(都電)', '').str.replace('(つくばＥＸＰ)', '')
    df['NearestStation'] = df['NearestStation'].str.replace('(千葉)',
                                                            '').str.replace(
        '(東京メトロ)', '').str.strip('()')
    return df


def extract_merge_key(df, col_name="AreaKey"):
    # 千代田区 + 飯田橋 = 千代田区飯田橋
    df[col_name] = df['Municipality'] + df['DistrictName']
    # published_land_price groupbyを使った特徴量を生成する際のkeyに利用
    # 5で切り出しているのはpublished_land_priceと一致させるため
    df[col_name] = df[col_name].str[:5]
    return df
