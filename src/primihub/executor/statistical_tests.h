#ifndef _STATISTICAL_TESTS_EXECUTOR_H_
#define _STATISTICAL_TESTS_EXECUTOR_H_

#include "src/primihub/common/common.h"
#include "aby3/sh3/Sh3Types.h"
#include "src/primihub/data_store/dataset.h"
#include "src/primihub/operator/aby3_operator.h"
#include "src/primihub/common/type.h"
#include "src/primihub/executor/statistics.h"

#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/result.h>
#include <cmath>

namespace primihub {

class MPCTTest : public MPCStatisticsOperator {
public:
  MPCTTest() {
    type_ = MPCStatisticsType::T_TEST;
  }
  
  virtual ~MPCTTest() {
    mpc_op_.reset();
  }

  retcode run(std::shared_ptr<primihub::Dataset> &dataset,
              const std::vector<std::string> &columns,
              const std::map<std::string, ColumnDtype> &col_dtype) override;
  
  retcode PlainTextDataCompute(std::shared_ptr<primihub::Dataset>& dataset,
      const std::vector<std::string>& columns,
      const std::map<std::string, ColumnDtype>& col_dtype,
      eMatrix<double>* result_data,
      eMatrix<double>* row_records) override;
  
  retcode CipherTextDataCompute(const eMatrix<double>& col_data,
                                const std::vector<std::string>& col_name,
                                const eMatrix<double>& row_records) override;
  
  retcode getResult(eMatrix<double> &result) override;

private:
  retcode computeLocalTTest(const eMatrix<double>& data1,
                           const eMatrix<double>& data2,
                           double* t_value,
                           double* df,
                           double* p_value);
  
  eMatrix<double> mpc_result_;
};

class MPCFTest : public MPCStatisticsOperator {
public:
  MPCFTest() {
    type_ = MPCStatisticsType::F_TEST;
  }
  
  virtual ~MPCFTest() {
    mpc_op_.reset();
  }

  retcode run(std::shared_ptr<primihub::Dataset> &dataset,
              const std::vector<std::string> &columns,
              const std::map<std::string, ColumnDtype> &col_dtype) override;
  
  retcode PlainTextDataCompute(std::shared_ptr<primihub::Dataset>& dataset,
      const std::vector<std::string>& columns,
      const std::map<std::string, ColumnDtype>& col_dtype,
      eMatrix<double>* result_data,
      eMatrix<double>* row_records) override;
  
  retcode CipherTextDataCompute(const eMatrix<double>& col_data,
                                const std::vector<std::string>& col_name,
                                const eMatrix<double>& row_records) override;
  
  retcode getResult(eMatrix<double> &result) override;

private:
  retcode computeLocalFTest(const eMatrix<double>& data1,
                           const eMatrix<double>& data2,
                           double* f_value,
                           double* df1,
                           double* df2,
                           double* p_value);
  
  eMatrix<double> mpc_result_;
};

class MPCChiSquareTest : public MPCStatisticsOperator {
public:
  MPCChiSquareTest() {
    type_ = MPCStatisticsType::CHI_SQUARE_TEST;
  }
  
  virtual ~MPCChiSquareTest() {
    mpc_op_.reset();
  }

  retcode run(std::shared_ptr<primihub::Dataset> &dataset,
              const std::vector<std::string> &columns,
              const std::map<std::string, ColumnDtype> &col_dtype) override;
  
  retcode PlainTextDataCompute(std::shared_ptr<primihub::Dataset>& dataset,
      const std::vector<std::string>& columns,
      const std::map<std::string, ColumnDtype>& col_dtype,
      eMatrix<double>* result_data,
      eMatrix<double>* row_records) override;
  
  retcode CipherTextDataCompute(const eMatrix<double>& col_data,
                                const std::vector<std::string>& col_name,
                                const eMatrix<double>& row_records) override;
  
  retcode getResult(eMatrix<double> &result) override;

private:
  retcode computeLocalChiSquare(const eMatrix<double>& observed,
                               const eMatrix<double>& expected,
                               double* chi2_value,
                               double* df,
                               double* p_value);
  
  eMatrix<double> mpc_result_;
};

class MPCCorrelation : public MPCStatisticsOperator {
public:
  MPCCorrelation() {
    type_ = MPCStatisticsType::CORRELATION;
  }
  
  virtual ~MPCCorrelation() {
    mpc_op_.reset();
  }

  retcode run(std::shared_ptr<primihub::Dataset> &dataset,
              const std::vector<std::string> &columns,
              const std::map<std::string, ColumnDtype> &col_dtype) override;
  
  retcode PlainTextDataCompute(std::shared_ptr<primihub::Dataset>& dataset,
      const std::vector<std::string>& columns,
      const std::map<std::string, ColumnDtype>& col_dtype,
      eMatrix<double>* result_data,
      eMatrix<double>* row_records) override;
  
  retcode CipherTextDataCompute(const eMatrix<double>& col_data,
                                const std::vector<std::string>& col_name,
                                const eMatrix<double>& row_records) override;
  
  retcode getResult(eMatrix<double> &result) override;

private:
  retcode computeLocalCorrelation(const eMatrix<double>& data1,
                                 const eMatrix<double>& data2,
                                 double* correlation,
                                 double* p_value);
  
  eMatrix<double> mpc_result_;
};

}; // namespace primihub

#endif