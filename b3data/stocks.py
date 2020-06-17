"""Nani."""

import glob
import numpy
import pandas as pd
from pkg_resources import resource_filename, Requirement

CLOSING = 'PREULT'
OPENING = 'PREABE'
MAX_PRICE = 'PREMAX'
MIN_PRICE = 'PREMIN'
MEAN_PRICE = 'PREMED'
VOLUME = 'VOLTOT'


class Stocks:
    """Nani."""

    def __init__(self, cod='PETR3', year=2014, start_month=1, period=6):
        """Nani."""
        path = resource_filename(Requirement.parse("b3data"), "data/COTAHIST_A" + str(year) + ".TXT.csv")
        files = glob.glob(path)

        for name in files:
            with open(name) as file:
                data_frame = pd.read_csv(file, low_memory=False)

            data_frame = data_frame.drop(data_frame.index[-1])
            data_frame = self._changing_column_data_type(data_frame)
            filtered = data_frame.loc[data_frame['CODNEG'] == cod]

            if (start_month + (period-1)) > 12:
                next_month = 12
                self.log("""Interval time surpassed the last month.
                            Last month will be 12""")
            else:
                next_month = start_month + (period-1)

            first_date = str(year) + '-' + str(start_month) + '-01'
            last_date = str(year) + '-' + str(next_month) + '-30'
            data_start = (filtered['DATA'] > first_date)
            data_end = (filtered['DATA'] <= last_date)
            stocks = filtered.loc[data_start & data_end]

            self.quotations = stocks

    def selected_fields(self, fields):
        """Nani."""
        if not fields:
            self.log('Argument fields cant be null')
        else:
            selected = self.quotations[[fields[0]]].values
            fields.pop(0)
            for i in fields:
                selected = numpy.column_stack((selected,
                                               self.quotations[[i]].values))

        return selected

    @staticmethod
    def interval_of_years(cod='PETR3', start_year=2014, end_year=2014,
                          start_month=1, period=5):
        quotations = numpy.array([])
        for year in range(start_year, end_year+1):
            stocks = Stocks(cod, year, start_month, period).selected_fields([CLOSING])
            for price in stocks:
                quotations = numpy.append(quotations, price)

        return quotations

    @staticmethod
    def log(message):
        """Nani."""
        print("[Stocks] " + message)

    @staticmethod
    def _changing_column_data_type(data_frame):
        """Nani."""
        data_frame['DATA'] = data_frame['DATA'].astype(str)
        data_frame['CODNEG'] = data_frame['CODNEG'].astype(str)
        data_frame['PREULT'] = (data_frame['PREULT']/100).astype(float)
        data_frame['PREABE'] = (data_frame['PREABE']/100).astype(float)
        data_frame['PREMAX'] = (data_frame['PREMAX']/100).astype(float)
        data_frame['PREMIN'] = (data_frame['PREMIN']/100).astype(float)
        data_frame['PREMED'] = (data_frame['PREMED']/100).astype(float)
        data_frame['VOLTOT'] = (data_frame['VOLTOT']/100).astype(float)
        data_frame['DATA'] = pd.to_datetime(data_frame['DATA'],
                                            format='%Y%m%d', errors='ignore')
        return data_frame
