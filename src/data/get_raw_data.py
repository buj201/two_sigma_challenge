import urllib2
import itertools
import os

def main(project_dir):
    ''' Get raw bike share data and saw in data/raw directory.
        See https://www.citibikenyc.com/system-data for description
        of raw Citi Bike Trip Histories dataset.

        Args:
            -project_dir: Path to project directory (os.path)
    '''

    baseurl = 'https://s3.amazonaws.com/tripdata/'

    years = [2016]
    months = [str(x).zfill(2) for x in range(1,13)]

    for year, month in itertools.product(years, months):
        print "Getting data for {}/{}...".format(month, year)
        filename = '{}{}-citibike-tripdata.zip'.format(year, month)
        request = urllib2.urlopen(baseurl + filename)
        with open(os.path.join(project_dir, 'data/raw/', filename), 'w') as f:
            f.write(request.read())

if __name__ == '__main__':
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    main(project_dir)
