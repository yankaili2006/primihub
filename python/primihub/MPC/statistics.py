import ph_secure_lib as ph_slib
from primihub.context import Context


class MPCJointStatistics:
    def __init__(self, protocol="ABY3"):
        cert_config = Context.cert_config
        root_ca_path = cert_config.get("root_ca_path", "")
        key_path = cert_config.get("key_path", "")
        cert_path = cert_config.get("cert_path", "")
        self.mpc_executor = ph_slib.MPCExecutor(
            Context.message, protocol, root_ca_path, key_path, cert_path
        )

    def max(self, input):
        """
        Input:
          input: local max data for each columns
        Output:
          max result
        """
        return self.mpc_executor.max(input)

    def min(self, input):
        """
        Input:
          input: local min data for each columns
          rows_of_columns: rows num of each columns
        Output:
          min result
        """
        return self.mpc_executor.min(input)

    def avg(self, input, rows_of_columns):
        """
        Input:
          input: local sum data for each columns
          rows_of_columns: rows num of each columns
        Output:
          avg result
        """
        return self.mpc_executor.avg(input, rows_of_columns)

    def sum(self, input):
        """
        Input:
          input: local sum data for each columns
        Output:
          sum result
        """
        return self.mpc_executor.sum(input)

    def t_test(self, data1, data2):
        """
        Perform federated T-test between two groups

        Input:
          data1: local data for group 1
          data2: local data for group 2
        Output:
          t-value, degrees of freedom, p-value
        """
        return self.mpc_executor.t_test(data1, data2)

    def f_test(self, data1, data2):
        """
        Perform federated F-test between two groups

        Input:
          data1: local data for group 1
          data2: local data for group 2
        Output:
          f-value, df1, df2, p-value
        """
        return self.mpc_executor.f_test(data1, data2)

    def chi_square_test(self, observed, expected=None):
        """
        Perform federated Chi-square test

        Input:
          observed: observed frequency counts
          expected: expected frequency counts (optional)
        Output:
          chi2-value, degrees of freedom, p-value
        """
        return self.mpc_executor.chi_square_test(observed, expected)

    def correlation(self, data1, data2):
        """
        Compute federated correlation between two variables

        Input:
          data1: local data for variable 1
          data2: local data for variable 2
        Output:
          correlation coefficient, p-value
        """
        return self.mpc_executor.correlation(data1, data2)
