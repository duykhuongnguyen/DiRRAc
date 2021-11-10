import numpy as np
import pandas as pd


class GermanCredit(object):
    """ Preprocess the german credit dataset """

    def __init__(self, d1_path, d2_path):
        """ Parameters

        Args:
            d1_path: path to initial dataset
            d2_path: path to correct dataset
            attr_dict: map from categorical attribute to numerical attribute
        """
        self.d1_path = d1_path
        self.d2_path = d2_path
        self.attr_dict = {'A11': 2, 'A12': 3, 'A13': 4, 'A14': 1,
                          'A30': 2, 'A31': 4, 'A32': 3, 'A33': 0, 'A34': 1,
                          'A40': 1, 'A41': 2, 'A42': 3, 'A43': 4, 'A44': 5, 'A45': 6, 'A46': 7, 'A47': 8, 'A48': 9, 'A49': 10, 'A410': 0,
                          'A61': 2, 'A62': 3, 'A63': 4, 'A64': 5, 'A65': 1,
                          'A71': 1, 'A72': 2, 'A73': 3, 'A74': 4, 'A75': 5,
                          'A91': 1, 'A92': 2, 'A93': 3, 'A94': 4, 'A95': 5,
                          'A101': 1, 'A102': 2, 'A103': 3,
                          'A121': 4, 'A122': 3, 'A123': 2, 'A124': 1,
                          'A141': 1, 'A142': 2, 'A143': 3,
                          'A151': 2, 'A152': 3, 'A153': 1,
                          'A171': 1, 'A172': 2, 'A173': 3, 'A174': 4,
                          'A191': 1, 'A192': 2,
                          'A201': 1, 'A202': 2,
                         }

    def read_data(self, path):
        """ Read data from file path

        Args:
            path: path to dataset file

        Returns:
        """
        data = np.zeros((1000, 20))
        labels = np.zeros(1000)

        with open(path, 'r') as f:
            lines = f.readlines()

        for i in range(len(lines)):
            line = lines[i][:-1].split(' ')
            for j in range(20):
                data[i][j] = self.attr_dict[line[j]] if line[j] in self.attr_dict else line[j]

            labels[i] = int(line[20]) if int(line[20]) < 2 else 2 - int(line[20])

        return data, labels

    def df_to_csv(self, data, labels, output_csv, save=True):
        """ Save to csv file """
        columns = ['GoodCustomer (label)', 'Status', 'Duration', 'History', 'Purpose', 'Credit amount',
                   'Savings account', 'Present employment', 'Installment rate', 'Personal status', 'Other debtors',
                   'Present residence', 'Property', 'Age', 'Other installment', 'Housing',
                   'Existing credits', 'Job', 'Number people', 'Telephone', 'Foreign worker']
        df = pd.DataFrame(columns=columns)
        df[columns[0]] = labels

        for i in range(1, len(columns)):
            df[columns[i]] = data[:, i - 1]

        if save:
            df.to_csv(output_csv, index=False)
        return df

    def normalize(self, data):
        """ Normalize continuous features to [0, 1]

        Args:
            data: data - numpy array shape (num samples, 20)

        Returns:
            data: data after normalize continuous features
        """
        data = (data - min(data)) / (max(data) - min(data))

        return data

    def one_hot_encode(self, data, length, zero):
        """ One hot encoding to all the instances

        Args:
            data: contains all instances

        Returns:
            encode: data after one hot encoding
        """
        encode_data = np.zeros((1000, length))
        for i in range(1000):
            sub = np.zeros(length)
            idx = data[i] if zero else data[i] - 1
            sub[int(idx)] = 1
            encode_data[i] = sub

        return encode_data

    def preprocess(self, save=False):
        """ Preprocess and get data """
        # Read and get raw csv
        d1, l1 = self.read_data(self.d1_path)
        d2, l2 = self.read_data(self.d2_path)
        df1 = self.df_to_csv(d1, l1, 'processed_data/german_raw.csv', save=save)
        df2 = self.df_to_csv(d2, l2, 'processed_data/german_raw_shift.csv', save=save)

        # Preprocess data
        continuous_cols = ['Duration', 'Credit amount', 'Installment rate', 'Present residence', 'Age', 'Existing credits', 'Number people']
        discrete_cols = ['Status', 'History']
        drop_cols = ['Purpose', 'Savings account', 'Present employment', 'Personal status', 'Other debtors', 'Property', 'Other installment', 'Housing', 'Job', 'Telephone', 'Foreign worker']

        df1 = df1.drop(columns=drop_cols)
        df2 = df2.drop(columns=drop_cols)
        if save:
            df1.to_csv('processed_data/german_processed.csv', index=False)
            df2.to_csv('processed_data/german_processed_shift.csv', index=False)
        # discrete_idx = {0: 4, 2: 5, 3: 11, 5: 5, 6: 5, 8: 5, 9: 3, 11: 4, 13: 3, 14: 3, 16: 4, 18: 2, 19: 2}
        # discrete_zeros = {0: 0, 2: 1, 3: 1, 5: 0, 6: 0, 8: 0, 9: 0, 11: 0, 13: 0, 14: 0, 16: 0, 18: 0, 19: 0}
        d1, d2 = np.zeros((1000, 17)), np.zeros((1000, 17))
        l1, l2 = df1['GoodCustomer (label)'].to_numpy(), df2['GoodCustomer (label)'].to_numpy()

        # Normalize continuous features
        for i in range(len(continuous_cols)):
            d1[:, i] = self.normalize(df1[continuous_cols[i]].to_numpy())
            d2[:, i] = self.normalize(df2[continuous_cols[i]].to_numpy())

        d1[:, 7:11] = self.one_hot_encode(df1['Status'].to_numpy(), 4, False)
        d1[:, 11:16] = self.one_hot_encode(df1['History'].to_numpy(), 5, True)
        d2[:, 7:11] = self.one_hot_encode(df2['Status'].to_numpy(), 4, False)
        d2[:, 11:16] = self.one_hot_encode(df2['History'].to_numpy(), 5, True)

        # Padding 1
        d1[:, 16] = np.ones(1000)
        d2[:, 16] = np.ones(1000)

        return d1, l1, d2, l2


class SBA(object):
    """ Preprocess SBA dataset """

    def __init__(self, fpath):
        """ Parameters

        Args:
            fpath: path to data csv file
        """
        self.df = pd.read_csv(fpath)
        self.df = self.df.drop(columns=['LoanNr_ChkDgt', 'Name', 'City', 'State', 'Zip', 'Bank', 'BankState', 'NAICS', 'ApprovalDate', 'NewExist', 'FranchiseCode', 'RevLineCr', 'LowDoc', 'ChgOffDate', 'DisbursementDate', 'DisbursementGross', 'BalanceGross', 'MIS_Status', 'daysterm', 'xx', 'Default'])
        self.df.rename({'Selected': 'Selected (label)'}, axis=1, inplace=True)

    def normalize(self, data):
        """ Normalize continuous features to [0, 1]

        Args:
            data: data - numpy array shape (num samples, 20)

        Returns:
            data: data after normalize continuous features
        """
        data = (data - min(data)) / (max(data) - min(data))

        return data

    def one_hot_encode(self, data, df_length, length, zero):
        """ One hot encoding to all the instances

        Args:
            data: contains all instances

        Returns:
            encode: data after one hot encoding
        """
        encode_data = np.zeros((df_length, length))
        for i in range(df_length):
            sub = np.zeros(length)
            idx = data[i] if zero else data[i] - 1
            sub[int(idx)] = 1
            encode_data[i] = sub

        return encode_data

    def preprocess(self, save=False):
        """ Split to d1 and d2 dataset """
        df1, df2 = self.df[self.df['ApprovalFY'] < 2006], self.df[self.df['ApprovalFY'] >= 2006]

        df1 = df1.drop(columns=['ApprovalFY'])
        df2 = df2.drop(columns=['ApprovalFY'])
        df1_length = len(df1)
        df2_length = len(df2)

        # Save data
        if save:
            df1.to_csv('processed_data/sba_processed.csv', index=False)
            df2.to_csv('processed_data/sba_processed_shift.csv', index=False)

        continuous_cols = ['Term', 'NoEmp', 'CreateJob', 'RetainedJob', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv', 'Portion']
        discrete_cols = ['UrbanRural', 'New', 'RealEstate', 'Recession']

        d1, d2 = np.zeros((df1_length, 18)), np.zeros((df2_length, 18))
        l1, l2 = df1['Selected (label)'].to_numpy(), df2['Selected (label)'].to_numpy()

        # Normalize continuous features
        for i in range(len(continuous_cols)):
            d1[:, i] = self.normalize(df1[continuous_cols[i]].to_numpy())
            d2[:, i] = self.normalize(df2[continuous_cols[i]].to_numpy())

        d1[:, 8:11] = self.one_hot_encode(df1['UrbanRural'].to_numpy(), df1_length, 3, True)
        d1[:, 11:13] = self.one_hot_encode(df1['New'].to_numpy(), df1_length, 2, True)
        d1[:, 13:15] = self.one_hot_encode(df1['RealEstate'].to_numpy(), df1_length, 2, True)
        d1[:, 15:17] = self.one_hot_encode(df1['Recession'].to_numpy(), df1_length, 2, True)

        d2[:, 8:11] = self.one_hot_encode(df2['UrbanRural'].to_numpy(), df2_length, 3, True)
        d2[:, 11:13] = self.one_hot_encode(df2['New'].to_numpy(), df2_length, 2, True)
        d2[:, 13:15] = self.one_hot_encode(df2['RealEstate'].to_numpy(), df2_length, 2, True)
        d2[:, 15:17] = self.one_hot_encode(df2['Recession'].to_numpy(), df2_length, 2, True)

        # Padding 1
        d1[:, 17] = np.ones(df1_length)
        d2[:, 17] = np.ones(df2_length)

        return d1, l1, d2, l2


class Student(object):
    """ Preprocess student dataset """

    def __init__(self, fpath):
        """ Parameters

        Args:
            fpath: path to data csv file
        """
        self.df = pd.read_csv(fpath, sep=';')

        # Attributes map
        self.df.sex = self.df.sex.map({'F': 0, 'M': 1})
        self.df.address = self.df.address.map({'U': 0, 'R': 1})
        self.df.famsize = self.df.famsize.map({'LE3': 0, 'GT3': 1})
        self.df.Pstatus = self.df.Pstatus.map({'T': 0, 'A': 1})
        self.df.Mjob = self.df.Mjob.map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
        self.df.Fjob = self.df.Fjob.map({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4})
        self.df.reason = self.df.reason.map({'home': 0, 'reputation': 1, 'course': 2, 'other': 3})
        self.df.guardian = self.df.guardian.map({'mother': 0, 'father': 1, 'other': 2})
        self.df.schoolsup = self.df.schoolsup.map({'yes': 0, 'no': 1})
        self.df.famsup = self.df.famsup.map({'yes': 0, 'no': 1})
        self.df.paid = self.df.paid.map({'yes': 0, 'no': 1})
        self.df.activities = self.df.activities.map({'yes': 0, 'no': 1})
        self.df.nursery = self.df.nursery.map({'yes': 0, 'no': 1})
        self.df.higher = self.df.higher.map({'yes': 0, 'no': 1})
        self.df.internet = self.df.internet.map({'yes': 0, 'no': 1})
        self.df.romantic = self.df.romantic.map({'yes': 0, 'no': 1})
        self.df.G3 = self.df.G3.map(lambda x: 1 if x >= 12 else 0)
        self.df.rename({'G3': 'G3 (label)'}, axis=1, inplace=True)

        # Drop columns
        self.df = self.df.drop(columns=['sex', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'failures', 'schoolsup', 'paid', 'activities', 'nursery', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc'])


    def normalize(self, data):
        """ Normalize continuous features to [0, 1]

        Args:
            data: data - numpy array shape (num samples, 20)

        Returns:
            data: data after normalize continuous features
        """
        data = (data - min(data)) / (max(data) - min(data))

        return data

    def one_hot_encode(self, data, df_length, length, zero):
        """ One hot encoding to all the instances

        Args:
            data: contains all instances

        Returns:
            encode: data after one hot encoding
        """
        encode_data = np.zeros((df_length, length))
        for i in range(df_length):
            sub = np.zeros(length)
            idx = data[i] if zero else data[i] - 1
            sub[int(idx)] = 1
            encode_data[i] = sub

        return encode_data

    def preprocess(self, save=False):
        """ Split to d1 and d2 dataset """
        df1, df2 = self.df[self.df['school'] == 'GP'], self.df[self.df['school'] == 'MS']

        df1 = df1.drop(columns=['school'])
        df2 = df2.drop(columns=['school'])
        df1_length = len(df1)
        df2_length = len(df2)

        # Save data
        if save:
            df1.to_csv('processed_data/student_processed.csv', index=False)
            df2.to_csv('processed_data/student_processed_shift.csv', index=False)

        continuous_cols = ['age', 'absences', 'G1', 'G2']
        discrete_cols = ['studytime', 'famsup', 'higher', 'internet', 'health']

        d1, d2 = np.zeros((df1_length, 20)), np.zeros((df2_length, 20))
        l1, l2 = df1['G3 (label)'].to_numpy(), df2['G3 (label)'].to_numpy()

        # Normalize continuous features
        for i in range(len(continuous_cols)):
            d1[:, i] = self.normalize(df1[continuous_cols[i]].to_numpy())
            d2[:, i] = self.normalize(df2[continuous_cols[i]].to_numpy())

        d1[:, 4:8] = self.one_hot_encode(df1['studytime'].to_numpy(), df1_length, 4, False)
        d1[:, 8:10] = self.one_hot_encode(df1['famsup'].to_numpy(), df1_length, 2, True)
        d1[:, 10:12] = self.one_hot_encode(df1['higher'].to_numpy(), df1_length, 2, True)
        d1[:, 12:14] = self.one_hot_encode(df1['internet'].to_numpy(), df1_length, 2, True)
        d1[:, 14:19] = self.one_hot_encode(df1['health'].to_numpy(), df1_length, 5, False)

        d2[:, 4:8] = self.one_hot_encode(df2['studytime'].to_numpy(), df2_length, 4, False)
        d2[:, 8:10] = self.one_hot_encode(df2['famsup'].to_numpy(), df2_length, 2, True)
        d2[:, 10:12] = self.one_hot_encode(df2['higher'].to_numpy(), df2_length, 2, True)
        d2[:, 12:14] = self.one_hot_encode(df2['internet'].to_numpy(), df2_length, 2, True)
        d2[:, 14:19] = self.one_hot_encode(df2['health'].to_numpy(), df2_length, 5, False)

        # Padding 1
        d1[:, 19] = np.ones(df1_length)
        d2[:, 19] = np.ones(df2_length)

        return d1, l1, d2, l2
