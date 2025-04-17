import os
import pandas as pd

#def read_as_parquet(csv_filename, create_parquet):
#
#    csv_filename = os.path.splitext(csv_filename)[0]
#
#    if create_parquet==True:
#
#        try:
#            data = pd.read_parquet("..\\03_Messdaten\\01_Messdaten_WZM\\" + csv_filename + ".parquet", engine='pyarrow')
#
#        except:
#
#            data = pd.read_csv("..\\03_Messdaten\\01_Messdaten_WZM\\" + csv_filename + ".csv", sep=';', header=0, index_col=0, parse_dates=True, squeeze=False, decimal =",")
#
#            data.to_parquet("..\\03_Messdaten\\01_Messdaten_WZM\\" + csv_filename + ".parquet")
#
#            data = pd.read_parquet("..\\03_Messdaten\\01_Messdaten_WZM\\" + csv_filename + ".parquet", engine='pyarrow')
#
#    else:
#        try:
#            data = pd.read_csv("..\\03_Messdaten\\01_Messdaten_WZM\\" + csv_filename + ".csv", sep=';', header=0, index_col=0, parse_dates=True, squeeze=False, decimal =",")
#        except:
#            print('The file does not exist')
#
#    return data

def read_as_parquet(csv_filename, create_parquet):

    file_path ="..\\03_Messdaten\\01_Messdaten_WZM\\{}".format(csv_filename)
    if create_parquet==True:
        if os.path.exists("{}.parquet".format(file_path)):
            data = pd.read_parquet("{}.parquet".format(file_path), engine='pyarrow')
        else:
            data = pd.read_csv("{}.csv".format(file_path), sep=';', header=0, index_col=0, parse_dates=True, squeeze=False, decimal =",")
            data.to_parquet("{}.parquet".format(file_path))
            data = pd.read_parquet("{}.parquet".format(file_path), engine='pyarrow')

    else:
        if os.path.exists("{}.csv".format(file_path)):
            data = pd.read_csv("{}.csv".format(file_path), sep=';', header=0, index_col=0, parse_dates=True, squeeze=False, decimal =",")
        else:
            print('The file does not exist')

    return data
