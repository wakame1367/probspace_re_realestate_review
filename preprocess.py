import numpy as np
from sklearn.preprocessing import LabelEncoder

from generate_feature import (built_year, walk_time, area1, area2, maguchi,
                              num_of_rooms,
                              nearest_station, ldk, total_floor_area_div_area,
                              total_floor_area_per_floor,
                              area_div_frontage, frontage_div_breadth)


def category_encode(df, target_cols):
    for col in target_cols:
        df[col] = df[col].fillna("NaN")
        le = LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
    return df


def preprocess(df):
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
