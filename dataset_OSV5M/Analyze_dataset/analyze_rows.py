import pandas as pd
df = pd.read_csv('/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/metadata_filtered/rest_filtered_v2.csv')
print(df['land_cover'].value_counts())
print(df['road_index'].describe())