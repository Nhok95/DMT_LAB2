maxi_lat = max(chlorophyllDF['latitude'].max(), temperatureDF['latitude'].max())
mini_lat = min(chlorophyllDF['latitude'].min(), temperatureDF['latitude'].min())
print('latitude min and max: [{}, {}]'.format(mini_lat, maxi_lat))   

maxi_lon = max(chlorophyllDF['longitude'].max(), temperatureDF['longitude'].max())
mini_lon = min(chlorophyllDF['longitude'].min(), temperatureDF['longitude'].min())
print('longitude min and max: [{}, {}]'.format(mini_lon, maxi_lon))   

latitude_coord = np.arange(-50, -35, 0.5)
latitude_coord

longitude_coord = np.arange(-70, -50, 0.5)
longitude_coord

lat = np.union1d(chlorophyllDF['latitude'].unique(), temperatureDF['latitude'].unique())
lon = np.union1d(chlorophyllDF['longitude'].unique(), temperatureDF['longitude'].unique())

mySquare = LineString([ (lat[0], lon[0]), (lat[0], lon[1]), (lat[1], lon[1]), (lat[1], lon[0]), (lat[0], lon[0])])
mySquare.bounds