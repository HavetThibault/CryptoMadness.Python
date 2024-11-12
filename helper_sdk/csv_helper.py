import csv


def get_csv_writer(file_descriptor):
    return csv.writer(file_descriptor, delimiter=',', quotechar='.', quoting=csv.QUOTE_MINIMAL)


def read_headers(filepath):
    with open(filepath, 'r', newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='.', quoting=csv.QUOTE_MINIMAL)
        return next(reader)
