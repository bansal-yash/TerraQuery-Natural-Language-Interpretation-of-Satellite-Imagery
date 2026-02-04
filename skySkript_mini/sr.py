import ee
ee.Initialize()

bbox = [22.23533398411717,60.4423105714194,22.237441462592425,60.443075551934406]
region = ee.Geometry.Rectangle(bbox)

col = ee.ImageCollection('COPERNICUS/S1_GRD') \
    .filterBounds(region) \
    .filterDate('2017-01-01','2017-12-31') \
    .filter(ee.Filter.eq('instrumentMode','IW')) \
    .select(['VV','VH'])

print('count:', col.size().getInfo())
img = col.median().clip(region)

# Stats (linear)
stats = img.reduceRegion(reducer=ee.Reducer.percentile([2,50,98]).combine(ee.Reducer.minMax(), '', True),
                         geometry=region, scale=10, bestEffort=True).getInfo()
print(stats)

# Convert to dB and apply focal median (approx)
img_db = img.log10().multiply(10)
img_db_filt = img_db.focal_median(30)  # radius in meters

# Export a small PNG via getThumbURL (scaled)
vis_params = {'min': -25, 'max': 0, 'bands': ['VV']}
thumb = img_db_filt.getThumbURL({'region': region.toGeoJSONString(), 'dimensions': 512, 'format': 'png', **vis_params})
print('thumb url:', thumb)
