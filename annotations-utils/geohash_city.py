import geohash
import geopandas as gpd
import folium
from polygon_geohasher.polygon_geohasher import geohash_to_polygon
import numpy as np


def create_map(lat, lng):
    # Create Geo Pandas DataFrame
    df = gpd.GeoDataFrame({'value': np.random.rand(9)})
    neighbors = geohash.neighbors(amsterdam)
    neighbors.append("u173zr")
    df['geohash'] = neighbors
    df['geometry'] = df['geohash'].apply(geohash_to_polygon)
    df.crs = {'init': 'epsg:4326'}
    m = folium.Map((lat, lng), zoom_start=12)
    folium.Choropleth(geo_data=df,
                      name='choropleth',
                      data=df,
                      columns=['geohash', 'value'],
                      key_on='feature.properties.geohash',
                      fill_color='YlGn',
                      fill_opacity=0.7,
                      line_opacity=0.2,
                      legend_name='asdf').add_to(m)

    m.save(f"geohashes.html")


if __name__ == "__main__":

    lat = 52.377956
    lng = 4.897070
    amsterdam = geohash.encode(lat, lng, 6) # u173zr
    decoded_location = geohash.decode(amsterdam)


    create_map(lat, lng)

