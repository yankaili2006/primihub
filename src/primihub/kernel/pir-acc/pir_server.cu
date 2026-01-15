#include "pir_server.cuh"
#include "pir_client.hpp"
#include "kernelutils.cuh"
#include <sigma.h>

using namespace std;
using namespace sigma;
using namespace sigma::util;
using namespace sigma::kernel_util;

PIRServer::PIRServer(const EncryptionParameters &enc_params,
                     const PirParams &pir_params)
    : enc_params_(enc_params), pir_params_(pir_params),
      is_db_preprocessed_(false) {

  context_ = make_shared<SIGMAContext>(enc_params, true);
  evaluator_ = make_unique<Evaluator>(*context_);
  encoder_ = make_unique<BatchEncoder>(*context_);
  //KernelProvider::copy(&device_enc_params_, &enc_params, 1);
  //KernelProvider::copy(&device_pir_params_, &pir_params, 1);//复制一份参数
  int logn = get_power_of_two(encoder_->slot_count());
  matrix_reps_index_map_ = HostArray<size_t>(encoder_->slot_count());

  // Copy from the matrix to the value vectors
  size_t row_size = encoder_->slot_count() >> 1;
  size_t m = encoder_->slot_count() << 1;
  uint64_t gen = 3;
  uint64_t pos = 1;
  for (size_t i = 0; i < row_size; i++) {
      // Position in normal bit order
      uint64_t index1 = (pos - 1) >> 1;
      uint64_t index2 = (m - pos - 1) >> 1;

      // Set the bit-reversed locations
      matrix_reps_index_map_[i] = safe_cast<size_t>(util::reverse_bits(index1, logn));
      matrix_reps_index_map_[row_size | i] = safe_cast<size_t>(util::reverse_bits(index2, logn));

      // Next primitive root
      pos *= gen;
      pos &= (m - 1);
  }
  Device_matrix_reps_index_map_ = matrix_reps_index_map_;
}

__global__ void printfdebug(uint64_t* plaintext){
    auto tid = blockDim.x * blockIdx.x + threadIdx.x;//线程索引
    printf("%d ",plaintext[tid]);
}

void PIRServer::preprocess_database() {

    uint32_t N = enc_params_.poly_modulus_degree();//4096
    uint64_t prod = 1;
    for (uint32_t i = 0; i < pir_params_.nvec.size(); i++) {
        prod *= pir_params_.nvec[i];
    }
    uint64_t matrix_plaintexts = prod;//7310
    auto result_ = make_unique<vector<Plaintext>>();
    result_->reserve(matrix_plaintexts);//预留内存并不分配

/*    Plaintext d_plaintexts;
    d_plaintexts.device_resize(matrix_plaintexts * N);*/
  if (!is_db_preprocessed_) {

      //printfdebug<<<4,1024>>>(device_db_->operator[](0).device_data());
      auto context_data_ptr = context_->get_context_data(context_->first_parms_id());
      auto &context_data = *context_data_ptr;
      auto &parms = context_data.parms();
      auto &coeff_modulus = parms.coeff_modulus();
      size_t coeff_count = parms.poly_modulus_degree();
      size_t coeff_modulus_size = coeff_modulus.size();

      //printf("%d ",db_->size());plaintexts数量
    #pragma unroll
    for (uint32_t i = 0; i < db_->size(); i++) {//plaintext形式每个都预处理一次

        evaluator_->preprocess_transform_to_ntt_inplace(db_->operator[](i), context_->first_parms_id());
        db_->operator[](i).resize(coeff_count * coeff_modulus_size);
        KernelProvider::retrieve(db_->operator[](i).data(),
                                 db_->operator[](i).device_data(), coeff_count * coeff_modulus_size);

        db_->operator[](i).parms_id() = context_->first_parms_id();

    }

    is_db_preprocessed_ = true;
  }
}

// Server takes over ownership of db and will free it when it exits
void PIRServer::set_database(unique_ptr<vector<Plaintext>> &&db) {
  if (!db) {
    throw invalid_argument("db cannot be null");
  }

  db_ = move(db);
  is_db_preprocessed_ = false;
}


__global__ void g_set_coefficients(uint64_t coeff_per_ptxt, uint32_t logt,
                                   uint8_t* bytes, uint64_t ele_size, uint64_t bytes_per_ptxt,
                                   uint64_t  coeff_per_element, uint64_t* plaintext,
                                   uint64_t num_of_plaintexts, uint64_t db_size,
                                   uint32_t N, uint32_t slot_count)
{//ele_size一个element多少字节
    auto tid = blockDim.x * blockIdx.x + threadIdx.x;//一个线程就负责一个plaintext
    //printf("tid:=%d\n",tid);
    uint64_t process_bytes = 0;
    uint64_t used = 0;
    if ((tid + 1) > num_of_plaintexts) {
        return;
    } else if ((tid + 1) == num_of_plaintexts) {
        process_bytes = db_size - tid * bytes_per_ptxt;
    } else {
        process_bytes = bytes_per_ptxt;
    }
    assert(process_bytes % ele_size == 0);//一个element一定填充整系数不连接

    uint64_t ele_in_chunk = process_bytes / ele_size;//需要处理数据 / 一个element数据

    used = ele_in_chunk * coeff_per_element;//使用了多少个系数

    int index = tid * bytes_per_ptxt;
    //#pragma unroll
    for (uint64_t ele = 0; ele < ele_in_chunk; ele++) {

        uint32_t room = logt;//阈值20 每次存取之后减少，存满20bit换成下一个系数继续存
        uint64_t *target = &plaintext[N * tid + coeff_per_element * ele];

        for (uint32_t i = 0; i < ele_size; i++) {
            //printf("i=%d:%d ",i,d_bytes[i]);
            uint8_t src = bytes[i + index + ele_size * ele];
            uint32_t rest = 8;
            while (rest) {
                if (room == 0) {
                    target++;//存下一个系数
                    room = logt;
                }
                uint32_t shift = rest;
                if (room < rest) {
                    shift = room;
                }
                *target = *target << shift;
                *target = *target | (src >> (8 - shift));
                src = src << shift;
                room -= shift;
                rest -= shift;
            }
        }
        *target = *target << room;

    }

    for (uint64_t j = N * tid + used; j < (coeff_per_ptxt); j++) {//将剩下的4096-3690填充
        plaintext[j] = 0;
    }

    for (uint64_t j = N * tid + coeff_per_ptxt; j < slot_count; j++) {//将剩下的4096-3690填充
        plaintext[j] = 1;
    }

}
__global__
void g_matrix_reps_index_map_place(uint64_t* values_matrix, uint64_t* destination,
                                   size_t values_matrix_size, const uint64_t* matrix_reps_index_map){
    auto tid = blockDim.x * blockIdx.x + threadIdx.x;

    *(destination + matrix_reps_index_map[tid]) = values_matrix[tid];
    __syncthreads();

}

void PIRServer::encode(uint64_t* values_matrix, uint64_t* destination, size_t values_matrix_size) const
{//加密values_matrix,destination
    auto &context_data = *context_->first_context_data();
    uint64_t blocknum = ceil(values_matrix_size / 1024.0);
    uint64_t coeff_power = get_power_of_two(values_matrix_size);
    g_matrix_reps_index_map_place<<<blocknum, 1024>>>(values_matrix, destination,
                                              values_matrix_size, Device_matrix_reps_index_map_.get());

    sigma::kernel_util::d_inverse_ntt_negacyclic_harvey(destination, 1, 1, coeff_power, *context_data.device_plain_ntt_tables());
    CHECK(cudaGetLastError());
}

void PIRServer::set_database(uint8_t* bytes,//bytes存在设备端
                             uint64_t ele_num, uint64_t ele_size) {//size_per_item = ele_size
//明文是小比特的数，但是加密之后数会变大，此时需要使用剩余数系统
    uint32_t logt = floor(log2(enc_params_.plain_modulus().value()));//20
    uint32_t N = enc_params_.poly_modulus_degree();//4096

    // number of FV plaintexts needed to represent all elements
    uint64_t num_of_plaintexts = pir_params_.num_of_plaintexts;//7282

    // number of FV plaintexts needed to create the d-dimensional matrix
    uint64_t prod = 1;
    for (uint32_t i = 0; i < pir_params_.nvec.size(); i++) {
        prod *= pir_params_.nvec[i];
    }
    uint64_t matrix_plaintexts = prod;//7310

    assert(num_of_plaintexts <= matrix_plaintexts);

    uint64_t ele_per_ptxt = pir_params_.elements_per_plaintext;//9
    uint64_t bytes_per_ptxt = ele_per_ptxt * ele_size;//9 * 1024

    uint64_t db_size = ele_num * ele_size;//65536*1024bytes 实际内容量

    uint64_t coeff_per_element = coefficients_per_element(logt, ele_size);

    uint64_t coeff_per_ptxt =
            ele_per_ptxt * coeff_per_element;//3690
    assert(coeff_per_ptxt <= N);

    cout << "Elements per plaintext: " << ele_per_ptxt << endl;//9
    cout << "Coeff per ptxt: " << coeff_per_ptxt << endl;//3690
    cout << "Bytes per plaintext: " << bytes_per_ptxt << endl;//9216

    uint32_t offset = 0;

    size_t grid_number = ceil(matrix_plaintexts / 1024.0);

    //uint64_t* d_plaintexts = KernelProvider::malloc<uint64_t>(matrix_plaintexts * N);
    Plaintext d_plaintexts;
    d_plaintexts.device_resize(matrix_plaintexts * N);

    auto &context_data = *context_->first_context_data();

    g_set_coefficients<<< grid_number, 1024 >>>(coeff_per_ptxt, logt, bytes, ele_size,
                                                bytes_per_ptxt, coeff_per_element, d_plaintexts.device_data(),
                                                num_of_plaintexts, db_size, N, pir_params_.slot_count);
    //cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
#ifdef DEBUG
    cout << "adding: " << matrix_plaintexts - current_plaintexts
       << " FV plaintexts of padding (equivalent to: "
       << (matrix_plaintexts - current_plaintexts) *
              elements_per_ptxt(logtp, N, ele_size)
       << " elements)" << endl;
#endif

    auto result_ = make_unique<vector<Plaintext>>();
    result_->reserve(matrix_plaintexts);//预留内存并不分配

    for(uint64_t i = 0; i < num_of_plaintexts; i++){

        Plaintext plain;
        plain.device_resize(N);

        encode(d_plaintexts.device_data() + i * N, plain.device_data(), N);
        result_->push_back(move(plain));
    }

    vector<uint64_t> padding(N, 1);
    uint64_t temp[N];
    std::copy(padding.begin(), padding.end(), temp);


    for (uint64_t i = 0; i < (matrix_plaintexts - num_of_plaintexts); i++) {
        Plaintext plain;
        plain.device_resize(N);
        KernelProvider::copy(plain.device_data(), temp, N);
        result_->push_back(plain);
    }

    set_database(move(result_));
}



void PIRServer::set_galois_key(uint32_t client_id, sigma::GaloisKeys galkey) {
  galoisKeys_[client_id] = galkey;
}

PirQuery PIRServer::deserialize_query(stringstream &stream) {
  PirQuery q;

  for (uint32_t i = 0; i < pir_params_.d; i++) {
    // number of ciphertexts needed to encode the index for dimension i
    // keeping into account that each ciphertext can encode up to
    // poly_modulus_degree indexes In most cases this is usually 1.
    uint32_t ctx_per_dimension =
        ceil((pir_params_.nvec[i] + 0.0) / enc_params_.poly_modulus_degree());

    vector<Ciphertext> cs;
    for (uint32_t j = 0; j < ctx_per_dimension; j++) {
      Ciphertext c;
      c.load(*context_, stream);
      cs.push_back(c);
    }

    q.push_back(cs);
  }

  return q;
}

int PIRServer::serialize_reply(PirReply &reply, stringstream &stream) {
  int output_size = 0;
  for (int i = 0; i < reply.size(); i++) {
    evaluator_->mod_switch_to_inplace(reply[i], context_->last_parms_id());
    output_size += reply[i].save(stream);
  }
  return output_size;
}

PirReply PIRServer::generate_reply(PirQuery &query, uint32_t client_id) {
//query 以ciphertext为元素的vector
//每个维度只有一个密文，但是一个密文长度不一定为16384

  vector<uint64_t> nvec = pir_params_.nvec;//维度设置为2
  uint64_t product = 1;

  for (uint32_t i = 0; i < nvec.size(); i++) {
    product *= nvec[i];//总的明文数量
  }

  auto coeff_count = enc_params_.poly_modulus_degree();//4096

  vector<Plaintext> *cur = db_.get();//db_数据以多项式的形式 plaintext 8192
  vector<Plaintext> intermediate_plain; // decompose....

  auto pool = MemoryManager::GetPool();

  int N = enc_params_.poly_modulus_degree();

  int logt = floor(log2(enc_params_.plain_modulus().value()));//明文的系数所用的比特数

  for (uint32_t i = 0; i < nvec.size(); i++) {//两个维度分别进行扩张
    cout << "Server: " << i + 1 << "-th recursion level started " << endl;

    vector<Ciphertext> expanded_query;

    uint64_t n_i = nvec[i];
    cout << "Server: n_i = " << n_i << endl;
    cout << "Server: expanding " << query[i].size() << " query ctxts" << endl;//一个query扩张 size得到的是1，该维度只用了一个密文就够
    //uint64_t* d_temp=KernelProvider::malloc<uint64_t>(4096);
    //Ciphertext d_temp;
    //d_temp.resize(4096)
    //KernelProvider::copy(d_temp.device_data(), query[i].data(), 4096);
    //kernel_util::
    for (uint32_t j = 0; j < query[i].size(); j++) {//进行一次扩张
      uint64_t total = N;
      if (j == query[i].size() - 1) {
        total = n_i % N;
      }
      cout << "-- expanding one query ctxt into " << total << " ctxts " << endl;//将一个问询密文扩成86个
      vector<Ciphertext> expanded_query_part =
          expand_query(query[i][j], total, client_id);
      expanded_query.insert(
          expanded_query.end(),
          std::make_move_iterator(expanded_query_part.begin()),
          std::make_move_iterator(expanded_query_part.end()));//expanded_query为非ntt
      expanded_query_part.clear();//临时存储
    }
/*    Ciphertext atempfortest;
    atempfortest.resize(*context_.get(), size_t(2), MemoryTypeHost);
    for(int i = 0; i < 8192*4; i++){
        atempfortest.data()[i]=1;
    }*/
    cout << "Server: expansion done " << endl;
    if (expanded_query.size() != n_i) {
      cout << " size mismatch!!! " << expanded_query.size() << ", " << n_i
           << endl;
    }
    //整个expanded_query已经具有device端的存储 数据库调为device
    // Transform expanded query to NTT, and ...
    #pragma unroll
    for (uint32_t jj = 0; jj < expanded_query.size(); jj++) {
        //expanded_query[jj].copy_to_device();
      evaluator_->sealpir_transform_to_ntt_inplace(expanded_query[jj]);//转为ntt
      //expanded_query[jj].retrieve_to_host();
    }

    // Transform plaintext to NTT. If database is pre-processed, can skip
    if ((!is_db_preprocessed_) || i > 0) {
    #pragma unroll
      for (uint32_t jj = 0; jj < cur->size(); jj++) {
        evaluator_->sealpir_transform_to_ntt_inplace((*cur)[jj],
                                             context_->first_parms_id());
          //KernelProvider::retrieve((*cur)[jj].data(), (*cur)[jj].device_data(), coeff_count);
      }
    }

    for (uint64_t k = 0; k < product; k++) {
      if ((*cur)[k].is_zero()) {
        cout << k + 1 << "/ " << product << "-th ptxt = 0 " << endl;
      }
    }

    product /= n_i;

    vector<Ciphertext> intermediateCtxts(product);
    Ciphertext temp;

    for (uint64_t k = 0; k < product; k++) {//二维数据库处理一个维度

      evaluator_->cu_multiply_plain(expanded_query[0], (*cur)[k],
                                 intermediateCtxts[k]);//一行对应一个expanded query，与该行的所有元素相乘

      for (uint64_t j = 1; j < n_i; j++) {
        evaluator_->cu_multiply_plain(expanded_query[j], (*cur)[k + j * product],
                                   temp);
        //temp.retrieve_to_host();
        evaluator_->cu_add_inplace(intermediateCtxts[k],
                                temp); // Adds to first component.
      }
    }

    for (uint32_t jj = 0; jj < intermediateCtxts.size(); jj++) {
      evaluator_->sealpir_transform_from_ntt_inplace(intermediateCtxts[jj]);//转换为系数，与数据库没有联系
      // print intermediate ctxts?
      // cout << "const term of ctxt " << jj << " = " <<
      // intermediateCtxts[jj][0] << endl;
    }

    if (i == nvec.size() - 1) {
        for(int number = 0; number < intermediateCtxts.size(); number++){
            intermediateCtxts[number].retrieve_to_host();
        }
      return intermediateCtxts;
    } else {
      intermediate_plain.clear();
      intermediate_plain.reserve(pir_params_.expansion_ratio * product);
      cur = &intermediate_plain;//cur转换到小的数据库

      for (uint64_t rr = 0; rr < product; rr++) {//扩展后的总的明文数量
        EncryptionParameters parms;
        if (pir_params_.enable_mswitching) {
          evaluator_->cu_mod_switch_to_inplace(intermediateCtxts[rr],
                                            context_->last_parms_id());//转换模数
          parms = context_->last_context_data()->parms();
        } else {
          parms = context_->first_context_data()->parms();
        }
        //intermediateCtxts[rr].copy_to_device();
        vector<Plaintext> plains =
            decompose_to_plaintexts(parms, intermediateCtxts[rr]);

        //KernelProvider::copy(plains.data()->device_data(), plains.data()->data(), plains.data()->coeff_count());
        for (uint32_t jj = 0; jj < plains.size(); jj++) {
            //KernelProvider::retrieve(plains[jj].data(), plains[jj].device_data(), plains[jj].coeff_count());
          intermediate_plain.emplace_back(plains[jj]);
        }
      }
      product = intermediate_plain.size(); // multiply by expansion rate.
    }
    cout << "Server: " << i + 1 << "-th recursion level finished " << endl;
    cout << endl;
  }
  cout << "reply generated!  " << endl;
  // This should never get here
  assert(0);
  vector<Ciphertext> fail(1);
  return fail;
}

void PIRServer::sealpir_apply_galois(
        const Ciphertext &encrypted, std::uint32_t galois_elt,
        const GaloisKeys &galois_keys, Ciphertext &destination)
{
    destination = encrypted;
    //evaluator_->sealpir_apply_galois_inplace(destination, galois_elt, galois_keys);
}

inline vector<Ciphertext> PIRServer::expand_query(const Ciphertext &encrypted,
                                                  uint32_t m,
                                                  uint32_t client_id) {//encrypted device
//密文，m=86即是维度 id即是用户标识 encrypted size为2 因为密文有两个元素组成
  GaloisKeys &galkey = galoisKeys_[client_id];

  // Assume that m is a power of 2. If not, round it to the next power of 2.
  uint32_t logm = ceil(log2(m));//m 86 logm 7
  Plaintext two("2");
  two.copy_to_device();

  vector<int> galois_elts;
  auto n = enc_params_.poly_modulus_degree();//4096
  if (logm > ceil(log2(n))) {
    throw logic_error("m > n is not allowed.");
  }
  for (int i = 0; i < ceil(log2(n)); i++) {//log 4096 = 12
    galois_elts.push_back((n + exponentiate_uint(2, i)) /
                          exponentiate_uint(2, i));//2^n+1 2-4097
  }//不以指针传参

  Ciphertext d_temp;
  d_temp.copy_device_from_host(encrypted);//d_temp的主机设备端都赋值

  vector<Ciphertext> temp;
  //temp.push_back(encrypted);//将query放入，一个维度的加密向量，一共一个密文，有两个元素所组成
  temp.push_back(d_temp);
  Ciphertext tempctxt;
  Ciphertext tempctxt_rotated;
  Ciphertext tempctxt_shifted;
  Ciphertext tempctxt_rotatedshifted;
  for (uint32_t i = 0; i < logm - 1; i++) {//7 - 1 每次扩展数量乘上2
    vector<Ciphertext> newtemp(temp.size() << 1);//乘二留出足够的空间
    //Ciphertext d_newtemp;
    //d_newtemp.resize(temp.size() << 1, MemoryTypeDevice);

    // temp[a] = (j0 = a (mod 2**i) ? ) : Enc(x^{j0 - a}) else Enc(0).  With
    // some scaling....
    int index_raw = (n << 1) - (1 << i);//8191（12位全部为1）
    int index = (index_raw * galois_elts[i]) % (n << 1);//4095

    for (uint32_t a = 0; a < temp.size(); a++) {//2 ciphertext密文指的是密文个数如(c0,c1)size为2

      evaluator_->apply_galois(temp[a], galois_elts[i], galkey,
                               tempctxt_rotated);//对于两个query分别进行操作 x^-1 x^-2 ......

      // cout << "rotate " <<
      // client.decryptor_->invariant_noise_budget(tempctxt_rotated) << ", ";

               //evaluator_->add(temp[a], tempctxt_rotated, newtemp[a]);
      evaluator_->cu_add(temp[a], tempctxt_rotated, newtemp[a]);


      multiply_power_of_X(temp[a], tempctxt_shifted, index_raw);

      // cout << "mul by x^pow: " <<
      // client.decryptor_->invariant_noise_budget(tempctxt_shifted) << ", ";

      multiply_power_of_X(tempctxt_rotated, tempctxt_rotatedshifted, index);

      // cout << "mul by x^pow: " <<
      // client.decryptor_->invariant_noise_budget(tempctxt_rotatedshifted) <<
      // ", ";

      // Enc(2^i x^j) if j = 0 (mod 2**i).
      evaluator_->cu_add(tempctxt_shifted, tempctxt_rotatedshifted,
                      newtemp[a + temp.size()]);
    }
    temp = newtemp;
  }
  // Last step of the loop
  vector<Ciphertext> newtemp(temp.size() << 1);
  int index_raw = (n << 1) - (1 << (logm - 1));
  int index = (index_raw * galois_elts[logm - 1]) % (n << 1);
  for (uint32_t a = 0; a < temp.size(); a++) {
    if (a >= (m - (1 << (logm - 1)))) { // corner case.

             //evaluator_->multiply_plain(temp[a], two,
                                         //newtemp[a]); // plain multiplication by 2.
      evaluator_->cu_multiply_plain(temp[a], two, newtemp[a]);
    } else {
      evaluator_->apply_galois(temp[a], galois_elts[logm - 1], galkey,
                               tempctxt_rotated);//temp[a] tempctxt_rotated device已经复制

                //evaluator_->add(temp[a], tempctxt_rotated, newtemp[a]);
      evaluator_->cu_add(temp[a], tempctxt_rotated, newtemp[a]);
      CHECK(cudaGetLastError());
      multiply_power_of_X(temp[a], tempctxt_shifted, index_raw);
      multiply_power_of_X(tempctxt_rotated, tempctxt_rotatedshifted, index);
      evaluator_->cu_add(tempctxt_shifted, tempctxt_rotatedshifted,
                      newtemp[a + temp.size()]);
    }
  }

  vector<Ciphertext>::const_iterator first = newtemp.begin();
  vector<Ciphertext>::const_iterator last = newtemp.begin() + m;
  vector<Ciphertext> newVec(first, last);

  return newVec;
}

inline void PIRServer::multiply_power_of_X(const Ciphertext &encrypted,
                                           Ciphertext &destination,
                                           uint32_t index) {

  auto coeff_mod_count = enc_params_.coeff_modulus().size() - 1;//多个模数
  auto coeff_count = enc_params_.poly_modulus_degree();//4096
  auto encrypted_count = encrypted.size();//2

  // First copy over.

  destination.copy_attributes(encrypted);
  destination.alloc_device_data(encrypted);

  // Prepare for destination
  // Multiply X^index for each ciphertext polynomial
  for (int i = 0; i < encrypted_count; i++) {//2


      sigma::kernel_util::d_negacyclic_shift_poly_coeffmod((uint64_t*)encrypted.device_data() + i * coeff_count * coeff_mod_count,
                                                       coeff_count, index, coeff_mod_count, enc_params_.device_coeff_modulus().get(),
                                                       destination.device_data() + i * coeff_count * coeff_mod_count);

  }

}

void PIRServer::simple_set(uint64_t index, Plaintext pt) {
  if (is_db_preprocessed_) {
    evaluator_->transform_to_ntt_inplace(pt, context_->first_parms_id());
  }
  db_->operator[](index) = pt;
}

Ciphertext PIRServer::simple_query(uint64_t index) {
  // There is no transform_from_ntt that takes a plaintext
  Ciphertext ct;
  Plaintext pt = db_->operator[](index);
  evaluator_->multiply_plain(one_, pt, ct);
  evaluator_->transform_from_ntt_inplace(ct);
  return ct;
}
void PIRServer::set_one_ct(Ciphertext one) {
  one_ = one;
  evaluator_->transform_to_ntt_inplace(one_);
}