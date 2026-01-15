#include "src/primihub/executor/statistical_tests.h"

#include <glog/logging.h>
#include <cmath>
#include <cstring>

namespace primihub {

#ifndef MPC_SOCKET_CHANNEL

// T-Test Implementation
retcode MPCTTest::PlainTextDataCompute(
    std::shared_ptr<primihub::Dataset>& dataset,
    const std::vector<std::string>& columns,
    const std::map<std::string, ColumnDtype>& col_dtype,
    eMatrix<double>* result_data,
    eMatrix<double>* row_records) {
  
  auto table = std::get<std::shared_ptr<arrow::Table>>(dataset->data);
  LOG(INFO) << "Schema of table is:" << table->schema()->ToString(true);
  
  // For T-test, we need at least 2 columns (two groups to compare)
  if (columns.size() < 2) {
    LOG(ERROR) << "T-test requires at least 2 columns for comparison";
    return retcode::FAIL;
  }
  
  result_data->resize(3, 1); // t-value, df, p-value
  row_records->resize(columns.size(), 1);
  
  // Extract data from first two columns
  std::vector<double> data1, data2;
  
  for (size_t i = 0; i < 2 && i < columns.size(); i++) {
    auto& col_name = columns[i];
    auto chunked_array = table->GetColumnByName(col_name);
    if (chunked_array.get() == nullptr) {
      LOG(ERROR) << "Can't get column value by column name " << col_name
                 << " from table.";
      return retcode::FAIL;
    }
    
    for (int chunk_idx = 0; chunk_idx < chunked_array->num_chunks(); chunk_idx++) {
      auto iter = col_dtype.find(col_name);
      const ColumnDtype &col_type = iter->second;
      
      if (col_type == ColumnDtype::INTEGER || col_type == ColumnDtype::LONG) {
        auto detected_type = table->schema()->GetFieldByName(col_name)->type();
        if (detected_type->id() == arrow::Type::INT32) {
          auto array = std::static_pointer_cast<arrow::Int32Array>(
              chunked_array->chunk(chunk_idx));
          for (int64_t j = 0; j < array->length(); j++) {
            if (i == 0) data1.push_back(array->Value(j));
            else data2.push_back(array->Value(j));
          }
        } else if (detected_type->id() == arrow::Type::INT64) {
          auto array = std::static_pointer_cast<arrow::Int64Array>(
              chunked_array->chunk(chunk_idx));
          for (int64_t j = 0; j < array->length(); j++) {
            if (i == 0) data1.push_back(array->Value(j));
            else data2.push_back(array->Value(j));
          }
        }
      } else if (col_type == ColumnDtype::DOUBLE) {
        auto array = std::static_pointer_cast<arrow::DoubleArray>(
            chunked_array->chunk(chunk_idx));
        for (int64_t j = 0; j < array->length(); j++) {
          if (i == 0) data1.push_back(array->Value(j));
          else data2.push_back(array->Value(j));
        }
      }
    }
    
    (*row_records)(i, 0) = (i == 0) ? data1.size() : data2.size();
  }
  
  // Compute local T-test statistics
  double t_value = 0, df = 0, p_value = 0;
  eMatrix<double> data1_mat(data1.size(), 1);
  eMatrix<double> data2_mat(data2.size(), 1);
  
  for (size_t i = 0; i < data1.size(); i++) data1_mat(i, 0) = data1[i];
  for (size_t i = 0; i < data2.size(); i++) data2_mat(i, 0) = data2[i];
  
  auto ret = computeLocalTTest(data1_mat, data2_mat, &t_value, &df, &p_value);
  if (ret != retcode::SUCCESS) {
    return ret;
  }
  
  (*result_data)(0, 0) = t_value;
  (*result_data)(1, 0) = df;
  (*result_data)(2, 0) = p_value;
  
  LOG(INFO) << "Local T-test computed: t=" << t_value << ", df=" << df << ", p=" << p_value;
  return retcode::SUCCESS;
}

retcode MPCTTest::CipherTextDataCompute(const eMatrix<double>& col_data,
                                       const std::vector<std::string>& col_name,
                                       const eMatrix<double>& row_records) {
  // For T-test, we need to securely compute statistics across parties
  // This would involve secure computation of means, variances, and t-statistic
  // For now, we'll implement a simplified version
  
  LOG(INFO) << "Starting secure T-test computation";
  
  // In a real implementation, we would:
  // 1. Securely compute means of both groups
  // 2. Securely compute variances
  // 3. Securely compute pooled variance
  // 4. Securely compute t-statistic
  // 5. Securely compute degrees of freedom
  // 6. Securely compute p-value
  
  // For this example, we'll just store the local results
  mpc_result_ = col_data;
  
  return retcode::SUCCESS;
}

retcode MPCTTest::run(std::shared_ptr<primihub::Dataset> &dataset,
                     const std::vector<std::string> &columns,
                     const std::map<std::string, ColumnDtype> &col_dtype) {
  eMatrix<double> col_data;
  eMatrix<double> rows_per_column;
  
  auto ret = PlainTextDataCompute(dataset, columns, col_dtype,
                                  &col_data, &rows_per_column);
  if (ret != retcode::SUCCESS) {
    LOG(ERROR) << "PlainTextDataCompute failed";
    return retcode::FAIL;
  }
  
  ret = CipherTextDataCompute(col_data, columns, rows_per_column);
  if (ret != retcode::SUCCESS) {
    LOG(ERROR) << "CipherTextDataCompute failed";
    return retcode::FAIL;
  }
  
  LOG(INFO) << "T-test computation completed";
  return retcode::SUCCESS;
}

retcode MPCTTest::getResult(eMatrix<double> &result) {
  if ((mpc_result_.rows() != result.rows()) ||
      (mpc_result_.cols() != result.cols())) {
    result.resize(mpc_result_.rows(), mpc_result_.cols());
    LOG(WARNING) << "Wrong shape of output matrix, reshape it.";
  }
  
  for (int i = 0; i < result.rows(); i++) {
    result(i, 0) = mpc_result_(i, 0);
  }
  
  return retcode::SUCCESS;
}

retcode MPCTTest::computeLocalTTest(const eMatrix<double>& data1,
                                   const eMatrix<double>& data2,
                                   double* t_value,
                                   double* df,
                                   double* p_value) {
  if (data1.rows() < 2 || data2.rows() < 2) {
    LOG(ERROR) << "Insufficient data for T-test";
    return retcode::FAIL;
  }
  
  // Compute means
  double mean1 = 0, mean2 = 0;
  for (int i = 0; i < data1.rows(); i++) mean1 += data1(i, 0);
  for (int i = 0; i < data2.rows(); i++) mean2 += data2(i, 0);
  mean1 /= data1.rows();
  mean2 /= data2.rows();
  
  // Compute variances
  double var1 = 0, var2 = 0;
  for (int i = 0; i < data1.rows(); i++) {
    double diff = data1(i, 0) - mean1;
    var1 += diff * diff;
  }
  for (int i = 0; i < data2.rows(); i++) {
    double diff = data2(i, 0) - mean2;
    var2 += diff * diff;
  }
  var1 /= (data1.rows() - 1);
  var2 /= (data2.rows() - 1);
  
  // Compute pooled standard error
  double pooled_var = ((data1.rows() - 1) * var1 + (data2.rows() - 1) * var2) /
                      (data1.rows() + data2.rows() - 2);
  double se = sqrt(pooled_var * (1.0/data1.rows() + 1.0/data2.rows()));
  
  // Compute t-statistic
  *t_value = (mean1 - mean2) / se;
  *df = data1.rows() + data2.rows() - 2;
  
  // Simplified p-value calculation (two-tailed)
  // In practice, use proper t-distribution CDF
  double abs_t = fabs(*t_value);
  *p_value = 2 * (1 - 0.5 * (1 + erf(abs_t / sqrt(2))));
  
  return retcode::SUCCESS;
}

// F-Test Implementation
retcode MPCFTest::PlainTextDataCompute(
    std::shared_ptr<primihub::Dataset>& dataset,
    const std::vector<std::string>& columns,
    const std::map<std::string, ColumnDtype>& col_dtype,
    eMatrix<double>* result_data,
    eMatrix<double>* row_records) {
  
  auto table = std::get<std::shared_ptr<arrow::Table>>(dataset->data);
  LOG(INFO) << "Schema of table is:" << table->schema()->ToString(true);
  
  // For F-test, we need at least 2 columns
  if (columns.size() < 2) {
    LOG(ERROR) << "F-test requires at least 2 columns";
    return retcode::FAIL;
  }
  
  result_data->resize(6, 1); // sum1, sum2, sum_sq1, sum_sq2, n1, n2
  row_records->resize(columns.size(), 1);
  
  // Extract data from two columns
  std::vector<double> data1, data2;
  
  for (size_t i = 0; i < 2 && i < columns.size(); i++) {
    auto& col_name = columns[i];
    auto chunked_array = table->GetColumnByName(col_name);
    if (chunked_array.get() == nullptr) {
      LOG(ERROR) << "Can't get column value by column name " << col_name
                 << " from table.";
      return retcode::FAIL;
    }
    
    for (int chunk_idx = 0; chunk_idx < chunked_array->num_chunks(); chunk_idx++) {
      auto iter = col_dtype.find(col_name);
      const ColumnDtype &col_type = iter->second;
      
      if (col_type == ColumnDtype::INTEGER || col_type == ColumnDtype::LONG) {
        auto detected_type = table->schema()->GetFieldByName(col_name)->type();
        if (detected_type->id() == arrow::Type::INT32) {
          auto array = std::static_pointer_cast<arrow::Int32Array>(
              chunked_array->chunk(chunk_idx));
          for (int64_t j = 0; j < array->length(); j++) {
            if (i == 0) data1.push_back(array->Value(j));
            else data2.push_back(array->Value(j));
          }
        } else if (detected_type->id() == arrow::Type::INT64) {
          auto array = std::static_pointer_cast<arrow::Int64Array>(
              chunked_array->chunk(chunk_idx));
          for (int64_t j = 0; j < array->length(); j++) {
            if (i == 0) data1.push_back(array->Value(j));
            else data2.push_back(array->Value(j));
          }
        }
      } else if (col_type == ColumnDtype::DOUBLE) {
        auto array = std::static_pointer_cast<arrow::DoubleArray>(
            chunked_array->chunk(chunk_idx));
        for (int64_t j = 0; j < array->length(); j++) {
          if (i == 0) data1.push_back(array->Value(j));
          else data2.push_back(array->Value(j));
        }
      }
    }
    
    (*row_records)(i, 0) = (i == 0) ? data1.size() : data2.size();
  }
  
  // Compute local statistics for F-test
  double sum1 = 0, sum_sq1 = 0;
  double sum2 = 0, sum_sq2 = 0;
  
  for (double val : data1) {
    sum1 += val;
    sum_sq1 += val * val;
  }
  
  for (double val : data2) {
    sum2 += val;
    sum_sq2 += val * val;
  }
  
  // Store intermediate results
  (*result_data)(0, 0) = sum1;
  (*result_data)(1, 0) = sum2;
  (*result_data)(2, 0) = sum_sq1;
  (*result_data)(3, 0) = sum_sq2;
  (*result_data)(4, 0) = data1.size();
  (*result_data)(5, 0) = data2.size();
  
  LOG(INFO) << "Local F-test statistics computed";
  return retcode::SUCCESS;
}

retcode MPCFTest::CipherTextDataCompute(const eMatrix<double>& col_data,
                                       const std::vector<std::string>& col_name,
                                       const eMatrix<double>& row_records) {
  LOG(INFO) << "Starting secure F-test computation";
  
  // Extract local statistics
  double local_sum1 = col_data(0, 0);
  double local_sum2 = col_data(1, 0);
  double local_sum_sq1 = col_data(2, 0);
  double local_sum_sq2 = col_data(3, 0);
  double local_n1 = col_data(4, 0);
  double local_n2 = col_data(5, 0);
  
  // Create matrices for MPC operations
  eMatrix<double> local_stats(6, 1);
  local_stats(0, 0) = local_sum1;
  local_stats(1, 0) = local_sum2;
  local_stats(2, 0) = local_sum_sq1;
  local_stats(3, 0) = local_sum_sq2;
  local_stats(4, 0) = local_n1;
  local_stats(5, 0) = local_n2;
  
  // Secure share local statistics
  sf64Matrix<D16> sh_stats[3];
  for (uint8_t i = 0; i < 3; i++)
    sh_stats[i].resize(6, 1);
  
  for (uint16_t i = 0; i < 3; i++) {
    if (i == party_id_)
      mpc_op_->createShares(local_stats, sh_stats[i]);
    else
      mpc_op_->createShares(sh_stats[i]);
  }
  
  // Sum statistics across all parties
  sf64Matrix<D16> sh_total_stats;
  sh_total_stats.resize(6, 1);
  sh_total_stats = sh_stats[0] + sh_stats[1] + sh_stats[2];
  
  // Reveal total statistics
  eMatrix<double> total_stats;
  for (uint16_t i = 0; i < 3; i++)
    if (i == party_id_)
      total_stats = mpc_op_->reveal(sh_total_stats);
    else
      mpc_op_->reveal(sh_total_stats, i);
  
  // Extract total statistics
  double total_sum1 = total_stats(0, 0);
  double total_sum2 = total_stats(1, 0);
  double total_sum_sq1 = total_stats(2, 0);
  double total_sum_sq2 = total_stats(3, 0);
  double total_n1 = total_stats(4, 0);
  double total_n2 = total_stats(5, 0);
  
  // Compute variances
  double var1 = (total_sum_sq1 - (total_sum1 * total_sum1) / total_n1) / (total_n1 - 1);
  double var2 = (total_sum_sq2 - (total_sum2 * total_sum2) / total_n2) / (total_n2 - 1);
  
  // Compute F-statistic (larger variance / smaller variance)
  double f_value = 0;
  double df1 = 0, df2 = 0;
  double p_value = 0;
  
  if (var1 > 0 && var2 > 0) {
    if (var1 >= var2) {
      f_value = var1 / var2;
      df1 = total_n1 - 1;
      df2 = total_n2 - 1;
    } else {
      f_value = var2 / var1;
      df1 = total_n2 - 1;
      df2 = total_n1 - 1;
    }
    
    // Compute p-value (two-tailed F-test)
    // Simplified calculation - in practice use F-distribution CDF
    if (df1 > 0 && df2 > 0) {
      // This is a simplified approximation
      double log_f = log(f_value);
      double z = (log_f - (1.0/df2 - 1.0/df1)) * sqrt(df1 * df2 / (df1 + df2));
      p_value = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))));
    }
  }
  
  // Store results
  mpc_result_.resize(4, 1);
  mpc_result_(0, 0) = f_value;
  mpc_result_(1, 0) = df1;
  mpc_result_(2, 0) = df2;
  mpc_result_(3, 0) = p_value;
  
  LOG(INFO) << "Secure F-test computed: F=" << f_value 
            << ", df1=" << df1 << ", df2=" << df2 << ", p=" << p_value;
  return retcode::SUCCESS;
}

retcode MPCFTest::run(std::shared_ptr<primihub::Dataset> &dataset,
                     const std::vector<std::string> &columns,
                     const std::map<std::string, ColumnDtype> &col_dtype) {
  eMatrix<double> col_data;
  eMatrix<double> rows_per_column;
  
  auto ret = PlainTextDataCompute(dataset, columns, col_dtype,
                                  &col_data, &rows_per_column);
  if (ret != retcode::SUCCESS) {
    return retcode::FAIL;
  }
  
  ret = CipherTextDataCompute(col_data, columns, rows_per_column);
  return ret;
}

retcode MPCFTest::getResult(eMatrix<double> &result) {
  if ((mpc_result_.rows() != result.rows()) ||
      (mpc_result_.cols() != result.cols())) {
    result.resize(mpc_result_.rows(), mpc_result_.cols());
  }
  
  for (int i = 0; i < result.rows(); i++) {
    result(i, 0) = mpc_result_(i, 0);
  }
  
  return retcode::SUCCESS;
}

// Chi-Square Test Implementation
retcode MPCChiSquareTest::PlainTextDataCompute(
    std::shared_ptr<primihub::Dataset>& dataset,
    const std::vector<std::string>& columns,
    const std::map<std::string, ColumnDtype>& col_dtype,
    eMatrix<double>* result_data,
    eMatrix<double>* row_records) {
  
  auto table = std::get<std::shared_ptr<arrow::Table>>(dataset->data);
  LOG(INFO) << "Schema of table is:" << table->schema()->ToString(true);
  
  // For Chi-square test of independence, we need at least 2 categorical columns
  if (columns.size() < 2) {
    LOG(ERROR) << "Chi-square test requires at least 2 columns";
    return retcode::FAIL;
  }
  
  // We'll implement a simple goodness-of-fit test for one column
  // For independence test, we'd need contingency table
  
  result_data->resize(10, 1); // Store observed frequencies (up to 10 categories)
  row_records->resize(2, 1);  // Store total count and number of categories
  
  // Extract data from first column
  auto& col_name = columns[0];
  auto chunked_array = table->GetColumnByName(col_name);
  if (chunked_array.get() == nullptr) {
    LOG(ERROR) << "Can't get column value by column name " << col_name
               << " from table.";
    return retcode::FAIL;
  }
  
  // Count frequencies (simplified - assuming integer categories 0-9)
  std::vector<int> freq_counts(10, 0);
  int total_count = 0;
  
  for (int chunk_idx = 0; chunk_idx < chunked_array->num_chunks(); chunk_idx++) {
    auto iter = col_dtype.find(col_name);
    const ColumnDtype &col_type = iter->second;
    
    if (col_type == ColumnDtype::INTEGER || col_type == ColumnDtype::LONG) {
      auto detected_type = table->schema()->GetFieldByName(col_name)->type();
      if (detected_type->id() == arrow::Type::INT32) {
        auto array = std::static_pointer_cast<arrow::Int32Array>(
            chunked_array->chunk(chunk_idx));
        for (int64_t j = 0; j < array->length(); j++) {
          int value = array->Value(j);
          if (value >= 0 && value < 10) {
            freq_counts[value]++;
            total_count++;
          }
        }
      } else if (detected_type->id() == arrow::Type::INT64) {
        auto array = std::static_pointer_cast<arrow::Int64Array>(
            chunked_array->chunk(chunk_idx));
        for (int64_t j = 0; j < array->length(); j++) {
          int64_t value = array->Value(j);
          if (value >= 0 && value < 10) {
            freq_counts[value]++;
            total_count++;
          }
        }
      }
    }
  }
  
  // Store observed frequencies
  for (int i = 0; i < 10; i++) {
    (*result_data)(i, 0) = freq_counts[i];
  }
  
  // Store metadata
  (*row_records)(0, 0) = total_count;
  (*row_records)(1, 0) = 10; // number of categories
  
  LOG(INFO) << "Local chi-square statistics computed: total=" << total_count;
  return retcode::SUCCESS;
}

retcode MPCChiSquareTest::CipherTextDataCompute(const eMatrix<double>& col_data,
                                               const std::vector<std::string>& col_name,
                                               const eMatrix<double>& row_records) {
  LOG(INFO) << "Starting secure Chi-square test computation";
  
  // Extract local observed frequencies
  eMatrix<double> local_observed(10, 1);
  for (int i = 0; i < 10; i++) {
    local_observed(i, 0) = col_data(i, 0);
  }
  
  double local_total = row_records(0, 0);
  double num_categories = row_records(1, 0);
  
  // Secure share observed frequencies
  sf64Matrix<D16> sh_observed[3];
  for (uint8_t i = 0; i < 3; i++)
    sh_observed[i].resize(10, 1);
  
  for (uint16_t i = 0; i < 3; i++) {
    if (i == party_id_)
      mpc_op_->createShares(local_observed, sh_observed[i]);
    else
      mpc_op_->createShares(sh_observed[i]);
  }
  
  // Sum observed frequencies across all parties
  sf64Matrix<D16> sh_total_observed;
  sh_total_observed.resize(10, 1);
  sh_total_observed = sh_observed[0] + sh_observed[1] + sh_observed[2];
  
  // Reveal total observed frequencies
  eMatrix<double> total_observed;
  for (uint16_t i = 0; i < 3; i++)
    if (i == party_id_)
      total_observed = mpc_op_->reveal(sh_total_observed);
    else
      mpc_op_->reveal(sh_total_observed, i);
  
  // Also need to share and sum total counts
  eMatrix<double> local_meta(2, 1);
  local_meta(0, 0) = local_total;
  local_meta(1, 0) = num_categories;
  
  sf64Matrix<D16> sh_meta[3];
  for (uint8_t i = 0; i < 3; i++)
    sh_meta[i].resize(2, 1);
  
  for (uint16_t i = 0; i < 3; i++) {
    if (i == party_id_)
      mpc_op_->createShares(local_meta, sh_meta[i]);
    else
      mpc_op_->createShares(sh_meta[i]);
  }
  
  sf64Matrix<D16> sh_total_meta;
  sh_total_meta.resize(2, 1);
  sh_total_meta = sh_meta[0] + sh_meta[1] + sh_meta[2];
  
  eMatrix<double> total_meta;
  for (uint16_t i = 0; i < 3; i++)
    if (i == party_id_)
      total_meta = mpc_op_->reveal(sh_total_meta);
    else
      mpc_op_->reveal(sh_total_meta, i);
  
  double grand_total = total_meta(0, 0);
  double categories = total_meta(1, 0);
  
  // Compute expected frequencies (uniform distribution)
  double expected = grand_total / categories;
  
  // Compute chi-square statistic
  double chi2 = 0;
  for (int i = 0; i < categories; i++) {
    double observed = total_observed(i, 0);
    if (expected > 0) {
      double diff = observed - expected;
      chi2 += (diff * diff) / expected;
    }
  }
  
  // Degrees of freedom
  double df = categories - 1;
  
  // Compute p-value (simplified)
  double p_value = 0;
  if (df > 0 && chi2 > 0) {
    // Simplified p-value calculation
    // In practice, use chi-square distribution CDF
    double z = (chi2 - df) / sqrt(2 * df);
    p_value = 1 - 0.5 * (1 + erf(z / sqrt(2)));
  }
  
  // Store results
  mpc_result_.resize(3, 1);
  mpc_result_(0, 0) = chi2;
  mpc_result_(1, 0) = df;
  mpc_result_(2, 0) = p_value;
  
  LOG(INFO) << "Secure chi-square test computed: chi2=" << chi2 
            << ", df=" << df << ", p=" << p_value;
  return retcode::SUCCESS;
}

retcode MPCChiSquareTest::run(std::shared_ptr<primihub::Dataset> &dataset,
                             const std::vector<std::string> &columns,
                             const std::map<std::string, ColumnDtype> &col_dtype) {
  eMatrix<double> col_data;
  eMatrix<double> rows_per_column;
  
  auto ret = PlainTextDataCompute(dataset, columns, col_dtype,
                                  &col_data, &rows_per_column);
  if (ret != retcode::SUCCESS) {
    return retcode::FAIL;
  }
  
  ret = CipherTextDataCompute(col_data, columns, rows_per_column);
  return ret;
}

retcode MPCChiSquareTest::getResult(eMatrix<double> &result) {
  if ((mpc_result_.rows() != result.rows()) ||
      (mpc_result_.cols() != result.cols())) {
    result.resize(mpc_result_.rows(), mpc_result_.cols());
  }
  
  for (int i = 0; i < result.rows(); i++) {
    result(i, 0) = mpc_result_(i, 0);
  }
  
  return retcode::SUCCESS;
}

// Correlation Implementation
retcode MPCCorrelation::PlainTextDataCompute(
    std::shared_ptr<primihub::Dataset>& dataset,
    const std::vector<std::string>& columns,
    const std::map<std::string, ColumnDtype>& col_dtype,
    eMatrix<double>* result_data,
    eMatrix<double>* row_records) {
  
  auto table = std::get<std::shared_ptr<arrow::Table>>(dataset->data);
  LOG(INFO) << "Schema of table is:" << table->schema()->ToString(true);
  
  // For correlation, we need at least 2 columns
  if (columns.size() < 2) {
    LOG(ERROR) << "Correlation requires at least 2 columns";
    return retcode::FAIL;
  }
  
  result_data->resize(3, 1); // sum_x, sum_y, sum_xy, sum_x2, sum_y2, count (for MPC)
  row_records->resize(6, 1); // We'll store 6 intermediate values
  
  // Extract data from first two columns
  std::vector<double> data1, data2;
  
  for (size_t i = 0; i < 2 && i < columns.size(); i++) {
    auto& col_name = columns[i];
    auto chunked_array = table->GetColumnByName(col_name);
    if (chunked_array.get() == nullptr) {
      LOG(ERROR) << "Can't get column value by column name " << col_name
                 << " from table.";
      return retcode::FAIL;
    }
    
    for (int chunk_idx = 0; chunk_idx < chunked_array->num_chunks(); chunk_idx++) {
      auto iter = col_dtype.find(col_name);
      const ColumnDtype &col_type = iter->second;
      
      if (col_type == ColumnDtype::INTEGER || col_type == ColumnDtype::LONG) {
        auto detected_type = table->schema()->GetFieldByName(col_name)->type();
        if (detected_type->id() == arrow::Type::INT32) {
          auto array = std::static_pointer_cast<arrow::Int32Array>(
              chunked_array->chunk(chunk_idx));
          for (int64_t j = 0; j < array->length(); j++) {
            if (i == 0) data1.push_back(array->Value(j));
            else data2.push_back(array->Value(j));
          }
        } else if (detected_type->id() == arrow::Type::INT64) {
          auto array = std::static_pointer_cast<arrow::Int64Array>(
              chunked_array->chunk(chunk_idx));
          for (int64_t j = 0; j < array->length(); j++) {
            if (i == 0) data1.push_back(array->Value(j));
            else data2.push_back(array->Value(j));
          }
        }
      } else if (col_type == ColumnDtype::DOUBLE) {
        auto array = std::static_pointer_cast<arrow::DoubleArray>(
            chunked_array->chunk(chunk_idx));
        for (int64_t j = 0; j < array->length(); j++) {
          if (i == 0) data1.push_back(array->Value(j));
          else data2.push_back(array->Value(j));
        }
      }
    }
  }
  
  // Check if both columns have the same number of observations
  if (data1.size() != data2.size()) {
    LOG(ERROR) << "Columns must have the same number of observations for correlation";
    return retcode::FAIL;
  }
  
  // Compute local statistics needed for Pearson correlation
  double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
  size_t n = data1.size();
  
  for (size_t i = 0; i < n; i++) {
    double x = data1[i];
    double y = data2[i];
    sum_x += x;
    sum_y += y;
    sum_xy += x * y;
    sum_x2 += x * x;
    sum_y2 += y * y;
  }
  
  // Store intermediate results for MPC computation
  (*result_data)(0, 0) = sum_x;
  (*result_data)(1, 0) = sum_y;
  (*result_data)(2, 0) = sum_xy;
  (*row_records)(0, 0) = sum_x2;
  (*row_records)(1, 0) = sum_y2;
  (*row_records)(2, 0) = n;
  
  LOG(INFO) << "Local correlation statistics computed: n=" << n 
            << ", sum_x=" << sum_x << ", sum_y=" << sum_y;
  return retcode::SUCCESS;
}

retcode MPCCorrelation::CipherTextDataCompute(const eMatrix<double>& col_data,
                                             const std::vector<std::string>& col_name,
                                             const eMatrix<double>& row_records) {
  LOG(INFO) << "Starting secure correlation computation";
  
  // Extract local statistics
  double local_sum_x = col_data(0, 0);
  double local_sum_y = col_data(1, 0);
  double local_sum_xy = col_data(2, 0);
  double local_sum_x2 = row_records(0, 0);
  double local_sum_y2 = row_records(1, 0);
  double local_n = row_records(2, 0);
  
  // Create matrices for MPC operations
  eMatrix<double> local_stats(6, 1);
  local_stats(0, 0) = local_sum_x;
  local_stats(1, 0) = local_sum_y;
  local_stats(2, 0) = local_sum_xy;
  local_stats(3, 0) = local_sum_x2;
  local_stats(4, 0) = local_sum_y2;
  local_stats(5, 0) = local_n;
  
  // Secure share local statistics
  sf64Matrix<D16> sh_stats[3];
  for (uint8_t i = 0; i < 3; i++)
    sh_stats[i].resize(6, 1);
  
  for (uint16_t i = 0; i < 3; i++) {
    if (i == party_id_)
      mpc_op_->createShares(local_stats, sh_stats[i]);
    else
      mpc_op_->createShares(sh_stats[i]);
  }
  
  // Sum statistics across all parties
  sf64Matrix<D16> sh_total_stats;
  sh_total_stats.resize(6, 1);
  sh_total_stats = sh_stats[0] + sh_stats[1] + sh_stats[2];
  
  // Reveal total statistics
  eMatrix<double> total_stats;
  for (uint16_t i = 0; i < 3; i++)
    if (i == party_id_)
      total_stats = mpc_op_->reveal(sh_total_stats);
    else
      mpc_op_->reveal(sh_total_stats, i);
  
  // Extract total statistics
  double total_sum_x = total_stats(0, 0);
  double total_sum_y = total_stats(1, 0);
  double total_sum_xy = total_stats(2, 0);
  double total_sum_x2 = total_stats(3, 0);
  double total_sum_y2 = total_stats(4, 0);
  double total_n = total_stats(5, 0);
  
  // Compute Pearson correlation coefficient
  double numerator = total_sum_xy - (total_sum_x * total_sum_y) / total_n;
  double denominator_x = total_sum_x2 - (total_sum_x * total_sum_x) / total_n;
  double denominator_y = total_sum_y2 - (total_sum_y * total_sum_y) / total_n;
  
  double correlation = 0;
  double p_value = 0;
  
  if (denominator_x > 0 && denominator_y > 0) {
    correlation = numerator / sqrt(denominator_x * denominator_y);
    
    // Compute p-value for correlation (two-tailed)
    if (total_n > 2) {
      double t_stat = correlation * sqrt((total_n - 2) / (1 - correlation * correlation));
      double df = total_n - 2;
      
      // Simplified p-value calculation using t-distribution
      // In practice, use proper t-distribution CDF
      double abs_t = fabs(t_stat);
      p_value = 2 * (1 - 0.5 * (1 + erf(abs_t / sqrt(2))));
    }
  }
  
  // Store results
  mpc_result_.resize(2, 1);
  mpc_result_(0, 0) = correlation;
  mpc_result_(1, 0) = p_value;
  
  LOG(INFO) << "Secure correlation computed: r=" << correlation << ", p=" << p_value;
  return retcode::SUCCESS;
}

retcode MPCCorrelation::run(std::shared_ptr<primihub::Dataset> &dataset,
                           const std::vector<std::string> &columns,
                           const std::map<std::string, ColumnDtype> &col_dtype) {
  eMatrix<double> col_data;
  eMatrix<double> rows_per_column;
  
  auto ret = PlainTextDataCompute(dataset, columns, col_dtype,
                                  &col_data, &rows_per_column);
  if (ret != retcode::SUCCESS) {
    return retcode::FAIL;
  }
  
  ret = CipherTextDataCompute(col_data, columns, rows_per_column);
  return ret;
}

retcode MPCCorrelation::getResult(eMatrix<double> &result) {
  if ((mpc_result_.rows() != result.rows()) ||
      (mpc_result_.cols() != result.cols())) {
    result.resize(mpc_result_.rows(), mpc_result_.cols());
  }
  
  for (int i = 0; i < result.rows(); i++) {
    result(i, 0) = mpc_result_(i, 0);
  }
  
  return retcode::SUCCESS;
}

#endif  // MPC_SOCKET_CHANNEL

};  // namespace primihub