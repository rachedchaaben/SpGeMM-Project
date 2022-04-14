#include <ginkgo/ginkgo.hpp>
// Add the fstream header to read from data from files.
#include <fstream>
// Add the C++ iostream header to output information to the console.
#include <benchmark/benchmark.h>
#include <omp.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "gtest/gtest.h"


template <typename ValueType, typename IndexType>
void swap(std::vector<std ::tuple<IndexType, IndexType, ValueType>>& result,
          IndexType a, IndexType b)
{
    auto t = result[a];
    result[a] = result[b];
    result[b] = t;
}

template <typename ValueType, typename IndexType>
IndexType partition(
    std::vector<std ::tuple<IndexType, IndexType, ValueType>>& result,
    IndexType low, IndexType high)
{
    auto pivot = result[high];
    auto i = (low - 1);
    for (IndexType j = low; j <= high - 1; j++) {
        if ((std::get<0>(result[j]) < std::get<0>(pivot)) ||
            ((std::get<0>(result[j]) == std::get<0>(pivot)) &&
             (std::get<1>(result[j]) <= std::get<1>(pivot)))) {
            i++;
            swap(result, i, j);
        }
    }
    swap(result, i + 1, high);
    return (i + 1);
}

template <typename ValueType, typename IndexType>
void quickSort(
    std::vector<std ::tuple<IndexType, IndexType, ValueType>>& result,
    IndexType low, IndexType high)
{
    if (low < high) {
        IndexType pi = partition(result, low, high);
#pragma omp task shared(result)  // firstprivate(result,low,pi)
        {
            quickSort(result, low, pi - 1);
        }
        //#pragma omp task firstprivate(arr, high,pi)
        {
            quickSort(result, pi + 1, high);
        }
    }
}


typedef std::pair<int, int> pair;

// Fixing CsrBuilder
namespace gko {
namespace matrix {
template <typename ValueType = default_precision, typename IndexType = int32>
class CsrBuilder {
public:
    Array<IndexType>& get_col_idx_array() { return matrix_->col_idxs_; }
    Array<ValueType>& get_value_array() { return matrix_->values_; }
    explicit CsrBuilder(Csr<ValueType, IndexType>* matrix) : matrix_{matrix} {}
    ~CsrBuilder() { matrix_->make_srow(); }
    CsrBuilder(const CsrBuilder&) = delete;
    CsrBuilder(CsrBuilder&&) = delete;
    CsrBuilder& operator=(const CsrBuilder&) = delete;
    CsrBuilder& operator=(CsrBuilder&&) = delete;

private:
    Csr<ValueType, IndexType>* matrix_;
};
}  // namespace matrix
}  // namespace gko

using mtx = gko::matrix::Csr<double, int>;
using size_type = std::size_t;
const auto exec = gko::ReferenceExecutor::create();
const auto omp_exec = gko::OmpExecutor::create();

// returns how many recursive splits are needed
template <typename SizeType>
SizeType recursive_splits(SizeType fast_mem_size, SizeType size_a,
                          SizeType size_b, SizeType num_rows_a,
                          SizeType num_cols_b)
{
    return ceil(
        log2(((double)std::min(num_rows_a * num_cols_b, size_a * size_b) /
              (double)fast_mem_size)));
}

// Binary search for the best split
template <typename ValueType, typename IndexType>
IndexType split_idx(IndexType* row_ptrs, ValueType num, IndexType start,
                    IndexType end)
{
    // Traverse the search space
    while (start <= end) {
        IndexType mid = (start + end) / 2;
        // case where the perfect split exists
        if (row_ptrs[mid] == num)
            return mid;
        else if (row_ptrs[mid] < num)
            start = mid + 1;
        else
            end = mid - 1;
    }
    // Return the split position for the most possible balanced split
    auto result =
        (num - row_ptrs[end] < row_ptrs[end + 1] - num) ? end : end + 1;
    return result;
}

// Binary search for the best split
template <typename ValueType, typename IndexType>
IndexType binary_search(const IndexType* row_ptrs, ValueType num,
                        IndexType start, IndexType end)
{
    // Traverse the search space
    while (start <= end) {
        IndexType mid = (start + end) / 2;
        // case where the perfect split exists
        if (row_ptrs[mid] == num)
            return mid;
        else if (row_ptrs[mid] < num)
            start = mid + 1;
        else
            end = mid - 1;
    }
    // Return the split position for the most possible balanced split
    return end + 1;
}

// generates offsets for a given matrix based on the number of recursive splits
// needed
template <typename IndexType>
void generate_offsets(std::vector<IndexType>& vect, IndexType* row_ptrs,
                      int rec_s, IndexType st, IndexType en)
{
    if ((en - st > 1) && (rec_s > 0)) {
        // Lower and upper bounds
        IndexType s = split_idx<gko::size_type, IndexType>(
            row_ptrs, (row_ptrs[en] + row_ptrs[st]) / 2, st, en);
        generate_offsets(vect, row_ptrs, rec_s - 1, st, s);
        vect.push_back(s);
        generate_offsets(vect, row_ptrs, rec_s - 1, s, en);
    }
}

// checks if there is an overlapping region between two matricies
template <typename ValueType, typename IndexType>
bool overlap(gko::matrix::Csr<ValueType, IndexType>* A,
             gko::matrix::Csr<ValueType, IndexType>* B_T, IndexType start_a,
             IndexType end_a, IndexType start_b, IndexType end_b)
{
    auto A_row_ptrs = A->get_row_ptrs();
    auto B_T_row_ptrs = B_T->get_row_ptrs();
    auto A_cols_index = A->get_col_idxs();
    auto B_T_cols_index = B_T->get_col_idxs();
    auto na = A->get_size()[0];
    auto nb = B_T->get_size()[0];
    for (int i = start_a; i < end_a; i++) {
        for (int j = start_b; j < end_b; j++) {
            if (A_cols_index[A_row_ptrs[i]] <=
                B_T_cols_index[B_T_row_ptrs[j]]) {
                if (A_cols_index[A_row_ptrs[i + 1] - 1] >=
                    B_T_cols_index[B_T_row_ptrs[j]])
                    return true;
            } else {
                if (B_T_cols_index[B_T_row_ptrs[j + 1] - 1] >=
                    A_cols_index[A_row_ptrs[i]])
                    return true;
            }
        }
    }
    return false;
}

//// Copied from reference implementation. Modified to work with rows and
/// columns ranges
//                             -----start-----
template <typename IndexType>
void prefix_sum(std::shared_ptr<const gko::ReferenceExecutor> exec,
                IndexType* counts, size_type num_entries)
{
    IndexType partial_sum{};
    for (size_type i = 0; i < num_entries; ++i) {
        auto nnz = counts[i];
        counts[i] = partial_sum;
        partial_sum += nnz;
    }
}

template <typename ValueType, typename IndexType>
void spgemm_accumulate_row2(std::map<IndexType, ValueType>& cols,
                            const gko::matrix::Csr<ValueType, IndexType>* a,
                            const gko::matrix::Csr<ValueType, IndexType>* b,
                            ValueType scale, size_type row, IndexType start_b,
                            IndexType end_b)
{
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_col_idxs = b->get_const_col_idxs();
    auto b_vals = b->get_const_values();
    for (size_type a_nz = a_row_ptrs[row];
         a_nz < size_type(a_row_ptrs[row + 1]); ++a_nz) {
        auto a_col = a_col_idxs[a_nz];
        auto a_val = a_vals[a_nz];
        auto b_row = a_col;
        auto col_start = binary_search<int, int>(
            b_col_idxs, start_b, b_row_ptrs[b_row], b_row_ptrs[b_row + 1] - 1);
        auto col_end = binary_search<int, int>(
            b_col_idxs, end_b, b_row_ptrs[b_row], b_row_ptrs[b_row + 1] - 1);
        for (size_type b_nz = col_start; b_nz < size_type(col_end); ++b_nz) {
            auto b_col = b_col_idxs[b_nz];
            auto b_val = b_vals[b_nz];
            cols[b_col - start_b] += scale * a_val * b_val;
        }
    }
}

template <typename ValueType, typename IndexType>
void unit_spgemm(std::shared_ptr<gko::ReferenceExecutor> exec,
                 const gko::matrix::Csr<ValueType, IndexType>* a,
                 const gko::matrix::Csr<ValueType, IndexType>* b,
                 std::vector<std ::tuple<IndexType, IndexType, ValueType>>& c,
                 IndexType start_a, IndexType end_a, IndexType start_b,
                 IndexType end_b)
{
    std::map<IndexType, ValueType> local_row_nzs;
    for (size_type a_row = start_a; a_row < end_a; ++a_row) {
        local_row_nzs.clear();
        spgemm_accumulate_row2(local_row_nzs, a, b, gko::one<ValueType>(),
                               a_row, start_b, end_b);
        // store result
        for (auto pair : local_row_nzs) {
            c.push_back(std::tuple<int, int, double>(
                a_row + 1, pair.first + start_b + 1, pair.second));
        }
    }
}
//                      -----end-----

template <typename ValueType, typename IndexType>
void unit_spgemm2(const gko::matrix::Csr<ValueType, IndexType>* a,
                  const gko::matrix::Csr<ValueType, IndexType>* b_T,
                  std::vector<std ::tuple<IndexType, IndexType, ValueType>>& c,
                  IndexType start_a, IndexType end_a, IndexType start_b,
                  IndexType end_b)
{
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto b_row_ptrs = b_T->get_const_row_ptrs();
    auto b_col_idxs = b_T->get_const_col_idxs();
    auto b_vals = b_T->get_const_values();
    for (size_type a_row = start_a; a_row < end_a; ++a_row) {
        for (size_type b_row = start_b; b_row < end_b; ++b_row) {
            auto a_nz = a_row_ptrs[a_row];
            auto b_nz = b_row_ptrs[b_row];
            ValueType result = 0;
            while (a_nz < a_row_ptrs[a_row + 1] &&
                   b_nz < b_row_ptrs[b_row + 1]) {
                auto a_col = a_col_idxs[a_nz];
                auto b_col = b_col_idxs[b_nz];
                bool eq = a_col == b_col;
                result += a_vals[a_nz] * b_vals[b_nz] * eq;
                bool eq2 = a_col < b_col;
                a_nz += eq2;
                a_nz += eq * (!eq2);
                b_nz += !eq2;
            }
            if (result != 0)
                c.push_back(
                    std::tuple<int, int, double>(a_row + 1, b_row + 1, result));
        }
    }
}

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const
    {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;
    }
};

template <typename ValueType, typename IndexType>
void unit_spgemm3(
    const gko::matrix::Csr<ValueType, IndexType>* a_T,
    const gko::matrix::Csr<ValueType, IndexType>* b,
    std::vector<std ::tuple<IndexType, IndexType, ValueType>>& result,
    IndexType start_a, IndexType end_a, IndexType start_b, IndexType end_b)
{
    std::unordered_set<std::pair<int, int>, pair_hash> keys;
    // std::vector<std ::tuple<IndexType, IndexType, ValueType>> matrix;
    auto a_T_row_ptrs = a_T->get_const_row_ptrs();
    auto a_T_col_idxs = a_T->get_const_col_idxs();
    auto a_vals = a_T->get_const_values();
    auto b_row_ptrs = b->get_const_row_ptrs();
    auto b_col_idxs = b->get_const_col_idxs();
    auto b_vals = b->get_const_values();

    ValueType matrix[end_a - start_a][end_b - start_b] = {0};
    for (size_type row = 0; row < b->get_size()[0]; ++row) {
        auto bs = binary_search<int, int>(b_col_idxs, start_b, b_row_ptrs[row],
                                          b_row_ptrs[row + 1] - 1);
        auto be = binary_search<int, int>(b_col_idxs, end_b, b_row_ptrs[row],
                                          b_row_ptrs[row + 1] - 1);
        auto as =
            binary_search<int, int>(a_T_col_idxs, start_a, a_T_row_ptrs[row],
                                    a_T_row_ptrs[row + 1] - 1);
        auto ae = binary_search<int, int>(
            a_T_col_idxs, end_a, a_T_row_ptrs[row], a_T_row_ptrs[row + 1] - 1);
        for (size_type a_idx = as; a_idx < ae; ++a_idx) {
            for (size_type b_idx = bs; b_idx < be; ++b_idx) {
                keys.insert(std::make_pair(a_T_col_idxs[a_idx] - start_a,
                                           b_col_idxs[b_idx] - start_b));
                auto col_val = b_col_idxs[b_idx];
                matrix[a_T_col_idxs[a_idx] - start_a]
                      [b_col_idxs[b_idx] - start_b] +=
                    a_vals[a_idx] * b_vals[b_idx];
            }
        }
    }
    for (auto elem : keys)
        result.push_back(std::tuple<int, int, double>(
            elem.first, elem.second, matrix[elem.first][elem.second]));
}


template <typename ValueType, typename IndexType>
void spgemm(gko::matrix::Csr<ValueType, IndexType>* a,
            gko::matrix::Csr<ValueType, IndexType>* b,
            gko::matrix::Csr<ValueType, IndexType>* c,
            IndexType recursive_splits)
{
    auto num_rows_c = a->get_size()[0];
    auto num_cols_c = b->get_size()[1];
    auto a_row_ptrs = a->get_row_ptrs();
    // auto a_T = gko::as<gko::matrix::Csr<double, int>>(a->transpose());
    auto b_T = gko::as<gko::matrix::Csr<double, int>>(b->transpose());
    auto b_T_row_ptrs = b_T->get_row_ptrs();

    std::vector<int> a_offsets{0};
    std::vector<int> b_offsets{0};
    generate_offsets<int>(a_offsets, a_row_ptrs, recursive_splits, 0,
                          num_rows_c);
    a_offsets.push_back((int)num_rows_c);
    generate_offsets<int>(b_offsets, b_T_row_ptrs, recursive_splits, 0,
                          num_cols_c);
    b_offsets.push_back((int)num_cols_c);

    // Calculation of all the small blocks
std::vector<std ::tuple<IndexType, IndexType, ValueType>> result;
#pragma omp parallel for collapse(2)
    for (auto row_idx = a_offsets.begin() + 1; row_idx < a_offsets.end();
         row_idx++) {
        for (auto col_idx = b_offsets.begin() + 1; col_idx < b_offsets.end();
             col_idx++) {
            // checking overlapping regions :
            if (overlap<double, int>(a, b_T.get(), *(row_idx - 1), *row_idx,
                                     *(col_idx - 1), *col_idx)) {
                std::vector<std ::tuple<IndexType, IndexType, ValueType>>
                    result1;
                /*unit_spgemm3<double, int>(a_T.get(), b, result1, *(row_idx -
                 *1), row_idx, *(col_idx - 1), *col_idx);/**/
                /*unit_spgemm2<double, int>(a, b_T.get(), result1, *(row_idx -
                 *1), row_idx, *(col_idx - 1), *col_idx);/**/
                unit_spgemm<double, int>(exec, a, b, result1, *(row_idx - 1),
                                         *row_idx, *(col_idx - 1),
                                         *col_idx); /**/
#pragma omp critical
                result.insert(result.end(), result1.begin(), result1.end());
            }
        }
    }
    std::sort(result.begin(), result.end());
    auto c_row_ptrs = c->get_row_ptrs();
    gko::matrix::CsrBuilder<double, int> c_builder{c};
    auto& c_col_idxs_array = c_builder.get_col_idx_array();
    auto& c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(result.size());
    c_vals_array.resize_and_reset(result.size());
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();
    c_row_ptrs[0] = 0;
    auto local_nnz = 0;
    auto c_nz = 0;
    for (std::vector<std::tuple<int, int, double>>::const_iterator i =
             result.begin();
         i != result.end(); ++i) {
        c_row_ptrs[std::get<1>(*i) - 1] += 1;
    }
    prefix_sum(exec, c_row_ptrs, num_rows_c + 1);

    for (std::vector<std::tuple<int, int, double>>::const_iterator i =
             result.begin();
         i != result.end(); ++i) {
        // c_row_ptrs[std::get<0>(*i)]+=1;
        c_col_idxs[c_nz] = std::get<1>(*i) - 1;
        c_vals[c_nz] = std::get<2>(*i);
        ++c_nz;
    }
    // gko::write(std::cout, c);
}

template <typename ValueType, typename IndexType>
void h_recursive_spgemm(
    gko::matrix::Csr<ValueType, IndexType>* a,
    gko::matrix::Csr<ValueType, IndexType>* b,
    gko::matrix::Csr<ValueType, IndexType>* a_T,
    gko::matrix::Csr<ValueType, IndexType>* b_T, IndexType h_recursive_splits,
    IndexType start_a, IndexType end_a, IndexType start_b, IndexType end_b,
    std::vector<std ::tuple<IndexType, IndexType, ValueType>>& result)
{
    auto num_cols = end_b - start_b;
    auto b_T_row_ptrs = b_T->get_row_ptrs();
    if (h_recursive_splits == 0 || num_cols < 2) {
        unit_spgemm<double, int>(exec, a, b, result, start_a, end_a, start_b,
                                 end_b); /**/
        /*unit_spgemm3<double, int>(a_T, b, result, start_a, end_a, start_b,
                                  end_b);/**/
    } else if (overlap<double, int>(a, b_T, start_a, end_a, start_b, end_b)) {
        auto b_idx = split_idx<gko::size_type, IndexType>(
            b_T_row_ptrs, (b_T_row_ptrs[end_b] + b_T_row_ptrs[start_b]) / 2,
            start_b, end_b);
        std::vector<std ::tuple<IndexType, IndexType, ValueType>> result1;
        std::vector<std ::tuple<IndexType, IndexType, ValueType>> result2;
#pragma omp task shared(result1)
        h_recursive_spgemm(a, b, a_T, b_T, h_recursive_splits - 1, start_a,
                           end_a, start_b, b_idx, result1);
#pragma omp task shared(result2)
        h_recursive_spgemm(a, b, a_T, b_T, h_recursive_splits - 1, start_a,
                           end_a, b_idx, end_b, result2);
#pragma omp taskwait
        result.insert(result.end(), result1.begin(), result1.end());
        result.insert(result.end(), result2.begin(), result2.end());
    }
}

template <typename ValueType, typename IndexType>
void v_recursive_spgemm(
    gko::matrix::Csr<ValueType, IndexType>* a,
    gko::matrix::Csr<ValueType, IndexType>* b,
    gko::matrix::Csr<ValueType, IndexType>* a_T,
    gko::matrix::Csr<ValueType, IndexType>* b_T, IndexType v_recursive_splits,
    IndexType h_recursive_splits, IndexType start_a, IndexType end_a,
    IndexType start_b, IndexType end_b,
    std::vector<std ::tuple<IndexType, IndexType, ValueType>>& result)
{
    auto num_rows = end_a - start_a;
    auto a_row_ptrs = a->get_row_ptrs();
    if (v_recursive_splits == 0 || num_rows < 2) {
        h_recursive_spgemm(a, b, a_T, b_T, h_recursive_splits, start_a, end_a,
                           start_b, end_b, result);
    } else if (overlap<double, int>(a, b_T, start_a, end_a, start_b, end_b)) {
        auto a_idx = split_idx<gko::size_type, IndexType>(
            a_row_ptrs, (a_row_ptrs[end_a] + a_row_ptrs[start_a]) / 2, start_a,
            end_a);
        std::vector<std ::tuple<IndexType, IndexType, ValueType>> result1;
        std::vector<std ::tuple<IndexType, IndexType, ValueType>> result2;
#pragma omp task shared(result1)
        v_recursive_spgemm(a, b, a_T, b_T, v_recursive_splits - 1,
                           h_recursive_splits, start_a, a_idx, start_b, end_b,
                           result1);
#pragma omp task shared(result2)
        v_recursive_spgemm(a, b, a_T, b_T, v_recursive_splits - 1,
                           h_recursive_splits, a_idx, end_a, start_b, end_b,
                           result2);
#pragma omp taskwait
        result.insert(result.end(), result1.begin(), result1.end());
        result.insert(result.end(), result2.begin(), result2.end());
    }
}

template <typename ValueType, typename IndexType>
void rec_spgemm(gko::matrix::Csr<ValueType, IndexType>* a,
                gko::matrix::Csr<ValueType, IndexType>* b,
                gko::matrix::Csr<ValueType, IndexType>* c,
                IndexType recursive_splits)
{
    auto a_T = gko::as<gko::matrix::Csr<double, int>>(a->transpose());
    auto b_T = gko::as<gko::matrix::Csr<double, int>>(b->transpose());
    std::vector<std ::tuple<IndexType, IndexType, ValueType>> result;
    v_recursive_spgemm<ValueType, IndexType>(
        a, b, a_T.get(), b_T.get(), recursive_splits, recursive_splits, 0,
        a->get_size()[0], 0, b->get_size()[1], result);
    std::sort(result.begin(), result.end());
    auto c_row_ptrs = c->get_row_ptrs();
    gko::matrix::CsrBuilder<double, int> c_builder{c};
    auto& c_col_idxs_array = c_builder.get_col_idx_array();
    auto& c_vals_array = c_builder.get_value_array();
    c_col_idxs_array.resize_and_reset(result.size());
    c_vals_array.resize_and_reset(result.size());
    auto c_col_idxs = c_col_idxs_array.get_data();
    auto c_vals = c_vals_array.get_data();
    c_row_ptrs[0] = 0;
    auto local_nnz = 0;
    auto c_nz = 0;
    for (std::vector<std::tuple<int, int, double>>::const_iterator i =
             result.begin();
         i != result.end(); ++i) {
        c_row_ptrs[std::get<1>(*i) - 1] += 1;
    }
    prefix_sum(exec, c_row_ptrs, a->get_size()[0] + 1);
    for (std::vector<std::tuple<int, int, double>>::const_iterator i =
             result.begin();
         i != result.end(); ++i) {
        c_col_idxs[c_nz] = std::get<1>(*i) - 1;
        c_vals[c_nz] = std::get<2>(*i);
        ++c_nz;
    }
}

//-----------------------------------------------------------------------------------------------------------
// -----------------------------------------------  Tests
// --------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

auto a = gko::read<mtx>(std::ifstream("../data/a2.mtx"), exec);
auto b = gko::read<mtx>(std::ifstream("../data/b2.mtx"), exec);
auto a_row_ptrs = a -> get_row_ptrs();
auto const_a_row_ptrs = a -> get_const_row_ptrs();
auto num_rows = a -> get_size()[0];

// testing the generation of number of recursive splits needed
TEST(example, recursive_splits)
{
    int m = 9;  // size of the fast memory
    auto res = recursive_splits<size_type>(m, a->get_num_stored_elements(),
                                           b->get_num_stored_elements(),
                                           a->get_size()[0], b->get_size()[1]);
    ASSERT_NEAR(res, 4, 1.0e-11);
}

// testing the generation of the split index the maximize the balance
TEST(example, split_idx)
{
    int start = 0;
    int end = 6;  // num_rows;
    auto s1 = split_idx<gko::size_type, int>(
        a_row_ptrs, (a_row_ptrs[0] + a_row_ptrs[num_rows]) / 2, 0, num_rows);
    auto s2 = split_idx<gko::size_type, int>(
        a_row_ptrs, (a_row_ptrs[0] + a_row_ptrs[6]) / 2, 0, 6);
    auto s3 = split_idx<gko::size_type, int>(
        a_row_ptrs, (a_row_ptrs[0] + a_row_ptrs[4]) / 2, 0, 4);
    ASSERT_NEAR(s1, 5, 1.0e-11);
    ASSERT_NEAR(s2, 3, 1.0e-11);
    ASSERT_NEAR(s3, 2, 1.0e-11);
}

// test for binary search function
TEST(example, binary_search)
{
    int start = 0;
    int end = 6;  // num_rows;
    auto s1 = binary_search<gko::size_type, int>(
        const_a_row_ptrs,
        (const_a_row_ptrs[0] + const_a_row_ptrs[num_rows]) / 2, 0, num_rows);
    auto s2 = binary_search<gko::size_type, int>(
        const_a_row_ptrs, (const_a_row_ptrs[0] + const_a_row_ptrs[6]) / 2, 0,
        6);
    auto s3 = binary_search<gko::size_type, int>(
        const_a_row_ptrs, (const_a_row_ptrs[0] + const_a_row_ptrs[4]) / 2, 0,
        4);
    ASSERT_NEAR(s1, 5, 1.0e-11);
    ASSERT_NEAR(s2, 4, 1.0e-11);
    ASSERT_NEAR(s3, 2, 1.0e-11);
}

TEST(example, generate_offsets)
{
    std::vector<int> result{3, 5, 7};
    std::vector<int> a_offsets{};
    generate_offsets<int>(a_offsets, a_row_ptrs, 2, 0, num_rows);
    auto idx1 = a_offsets.begin();
    auto idx2 = result.begin();

    for (int i = 0; i < 3; i++) {
        ASSERT_NEAR(*idx1, *idx2, 1.0e-11);
        idx1++;
        idx2++;
    }
}

TEST(example, spgemm)
{
    auto A = gko::read<mtx>(std::ifstream("../data/a9.mtx"), exec);
    auto B = gko::read<mtx>(std::ifstream("../data/b9.mtx"), exec);
    auto C = mtx::create(exec, gko::dim<2>{A->get_size()[0], B->get_size()[1]});
    auto C2 =
        mtx::create(exec, gko::dim<2>{A->get_size()[0], B->get_size()[1]});
    auto a_row_ptrs = A->get_row_ptrs();
    auto a_T = gko::as<gko::matrix::Csr<double, int>>(A->transpose());
    auto b_T = gko::as<gko::matrix::Csr<double, int>>(B->transpose());
    auto b_T_row_ptrs = b_T->get_row_ptrs();
    auto num_cols = B->get_size()[1];
    A->apply(B.get(), C.get());
    spgemm<double, int>(A.get(), B.get(), C2.get(), 2);
    auto c_row_ptrs = C->get_row_ptrs();
    auto c_col_idxs = C->get_col_idxs();
    auto c_vals = C->get_values();   
    auto c2_row_ptrs = C2->get_row_ptrs();
    auto c2_col_idxs = C2->get_col_idxs();
    auto c2_vals = C2->get_values();
    auto num_entries_c=C->get_num_stored_elements();
    auto num_entries_c2=C2->get_num_stored_elements();
    ASSERT_NEAR(num_entries_c,num_entries_c2, 1.0e-11);
    for (auto row = 0; row < C->get_size()[0]; ++row) {
        for (auto ent = c_row_ptrs[row]; ent < c_row_ptrs[row + 1]; ++ent) {
            auto col = c_col_idxs[ent]; auto col2 = c2_col_idxs[ent];;
            auto val = c_vals[ent]; auto val2 = c2_vals[ent];
            ASSERT_NEAR(col,col2, 1.0e-11);
            ASSERT_NEAR(val,val2, 1.0e-11);
        }
    }
}

TEST(example, v_recursive_spgemm)
{
    auto A = gko::read<mtx>(std::ifstream("../data/a5.mtx"), exec);
    auto B = gko::read<mtx>(std::ifstream("../data/b5.mtx"), exec);
    auto C = mtx::create(exec, gko::dim<2>{A->get_size()[0], B->get_size()[1]});
    auto a_row_ptrs = A->get_row_ptrs();
    auto a_T = gko::as<gko::matrix::Csr<double, int>>(A->transpose());
    auto b_T = gko::as<gko::matrix::Csr<double, int>>(B->transpose());
    auto b_T_row_ptrs = b_T->get_row_ptrs();
    auto num_cols = B->get_size()[1];
    A->apply(B.get(), C.get());
    std::vector<std ::tuple<int, int, double>> result;
    v_recursive_spgemm<double, int>(A.get(), B.get(), a_T.get(), b_T.get(), 3,
                                    3, 0, A->get_size()[0], 0, B->get_size()[1],
                                    result);
    std::sort(result.begin(), result.end());
    auto entry = result.begin();

    auto c_row_ptrs = C->get_row_ptrs();
    auto c_col_idxs = C->get_col_idxs();
    auto c_vals = C->get_values();

    for (auto row = 0; row < C->get_size()[0]; ++row) {
        for (auto ent = c_row_ptrs[row]; ent < c_row_ptrs[row + 1]; ++ent) {
            auto col = c_col_idxs[ent];
            auto val = c_vals[ent];
            ASSERT_NEAR(row + 1, std::get<0>(*entry), 1.0e-11);
            ASSERT_NEAR(col + 1, std::get<1>(*entry), 1.0e-11);
            ASSERT_NEAR(val, std::get<2>(*entry), 1.0e-11);
            entry++;
        }
    }
}

TEST(example, rec_spgemm)
{
    auto A = gko::read<mtx>(std::ifstream("../data/a5.mtx"), exec);
    auto B = gko::read<mtx>(std::ifstream("../data/b5.mtx"), exec);
    auto C = mtx::create(exec, gko::dim<2>{A->get_size()[0], B->get_size()[1]});
    auto C2 =
        mtx::create(exec, gko::dim<2>{A->get_size()[0], B->get_size()[1]});
    auto a_row_ptrs = A->get_row_ptrs();
    auto a_T = gko::as<gko::matrix::Csr<double, int>>(A->transpose());
    auto b_T = gko::as<gko::matrix::Csr<double, int>>(B->transpose());
    auto b_T_row_ptrs = b_T->get_row_ptrs();
    auto num_cols = B->get_size()[1];
    A->apply(B.get(), C.get());
    rec_spgemm<double, int>(A.get(), B.get(), C2.get(), 2);

    auto c_row_ptrs = C->get_row_ptrs();
    auto c_col_idxs = C->get_col_idxs();
    auto c_vals = C->get_values();   
    auto c2_row_ptrs = C2->get_row_ptrs();
    auto c2_col_idxs = C2->get_col_idxs();
    auto c2_vals = C2->get_values();
    auto num_entries_c=C->get_num_stored_elements();
    auto num_entries_c2=C2->get_num_stored_elements();
    ASSERT_NEAR(num_entries_c,num_entries_c2, 1.0e-11);
    for (auto row = 0; row < C->get_size()[0]; ++row) {
        for (auto ent = c_row_ptrs[row]; ent < c_row_ptrs[row + 1]; ++ent) {
            auto col = c_col_idxs[ent]; auto col2 = c2_col_idxs[ent];;
            auto val = c_vals[ent]; auto val2 = c2_vals[ent];
            ASSERT_NEAR(col,col2, 1.0e-11);
            ASSERT_NEAR(val,val2, 1.0e-11);
        }
    }
}

//-----------------------------------------------------------------------------------------------------------
// ----------------------------------------------- Benchmarks
// -----------------------------------------------
//-----------------------------------------------------------------------------------------------------------

auto A = gko::read<mtx>(std::ifstream("../data/a9.mtx"), omp_exec);
auto B = gko::read<mtx>(std::ifstream("../data/b9.mtx"), omp_exec);
auto C = mtx::create(omp_exec, gko::dim<2>{A->get_size()[0], B->get_size()[1]});
auto a_T = gko::as<gko::matrix::Csr<double, int>>(A->transpose());
auto b_T = gko::as<gko::matrix::Csr<double, int>>(B->transpose());

static void BM_OLD_SpGEMM(benchmark::State& state)
{
    for (auto _ : state) {
        auto A = gko::read<mtx>(
            std::ifstream("../data/a" + std::to_string(state.range(0)) +
                          ".mtx"),
            omp_exec);
        auto B = gko::read<mtx>(
            std::ifstream("../data/b" + std::to_string(state.range(0)) +
                          ".mtx"),
            omp_exec);
        auto C = mtx::create(omp_exec,
                             gko::dim<2>{A->get_size()[0], B->get_size()[1]});
        A->apply(B.get(), C.get());
    }
}
BENCHMARK(BM_OLD_SpGEMM)->Arg(4)->Arg(5)->Arg(6)->Arg(7)->Arg(8)->Arg(19);

static void BM_SpGEMM(benchmark::State& state)
{
    for (auto _ : state) {
        auto A = gko::read<mtx>(
            std::ifstream("../data/a" + std::to_string(state.range(0)) +
                          ".mtx"),
            omp_exec);
        auto B = gko::read<mtx>(
            std::ifstream("../data/b" + std::to_string(state.range(0)) +
                          ".mtx"),
            omp_exec);
        auto C = mtx::create(omp_exec,
                             gko::dim<2>{A->get_size()[0], B->get_size()[1]});
        spgemm<double, int>(A.get(), B.get(), C.get(), 2);
    }
}
BENCHMARK(BM_SpGEMM)->Arg(4)->Arg(5)->Arg(6)->Arg(7)->Arg(8)->Arg(19);

static void BM_REC_SpGEMM(benchmark::State& state)
{
    for (auto _ : state) {
        auto A = gko::read<mtx>(
            std::ifstream("../data/a" + std::to_string(state.range(0)) +
                          ".mtx"),
            omp_exec);
        auto B = gko::read<mtx>(
            std::ifstream("../data/b" + std::to_string(state.range(0)) +
                          ".mtx"),
            omp_exec);
        auto C = mtx::create(omp_exec,
                             gko::dim<2>{A->get_size()[0], B->get_size()[1]});
        auto a_T = gko::as<gko::matrix::Csr<double, int>>(A->transpose());
        auto b_T = gko::as<gko::matrix::Csr<double, int>>(B->transpose());
#pragma omp parallel
       #pragma omp single
        rec_spgemm<double,int>(A.get(),B.get(),C.get(),4);
    }
}
BENCHMARK(BM_REC_SpGEMM)->Arg(4)->Arg(5)->Arg(6)->Arg(7)->Arg(8)->Arg(19);

/*

static void BM_UNIT_SPGEMM1(benchmark::State& state)
{
    std::vector<std ::tuple<int, int, double>> result;
    for (auto _ : state ){
    unit_spgemm<double, int>(exec, A.get(), B.get(), result, 550+state.range(0),
650+state.range(0),200+state.range(0), 300+state.range(0));
    }
}
BENCHMARK(BM_UNIT_SPGEMM1)->Arg(23)->Arg(321)->Arg(11)->Arg(1000);

static void BM_UNIT_SPGEMM2(benchmark::State& state)
{
    std::vector<std ::tuple<int, int, double>> result;
    for (auto _ : state ){
    unit_spgemm2<double, int>(A.get(), b_T.get(), result,550+state.range(0),
650+state.range(0),200+state.range(0), 300+state.range(0));
    }
}
BENCHMARK(BM_UNIT_SPGEMM2)->Arg(23)->Arg(321)->Arg(11)->Arg(1000);

static void BM_UNIT_SPGEMM3(benchmark::State& state)
{
    std::vector<std ::tuple<int, int, double>> result;
    for (auto _ : state ){
    unit_spgemm3<double, int>(a_T.get(), B.get(), result, 550+state.range(0),
650+state.range(0),200+state.range(0), 300+state.range(0));
    }
}
BENCHMARK(BM_UNIT_SPGEMM3)->Arg(23)->Arg(321)->Arg(11)->Arg(1000);

static void BM_OVERLAP(benchmark::State& state)
{
    auto b_col_idxs = B->get_const_col_idxs();
    auto a_T_col_idxs = a_T->get_const_col_idxs();
    auto a_T_row_ptrs = a_T->get_const_row_ptrs();
    auto b_row_ptrs = b->get_const_row_ptrs();
    for (auto _ : state){
       for (size_type row = 0; row < b->get_size()[0]; ++row) {
        auto bs = binary_search<int, int>(
            b_col_idxs, 200, b_row_ptrs[row], b_row_ptrs[row + 1] - 1);
        auto be = binary_search<int, int>(
            b_col_idxs, 900, b_row_ptrs[row], b_row_ptrs[row + 1] - 1);
        auto as = binary_search<int, int>(
            a_T_col_idxs, 155, a_T_row_ptrs[row], a_T_row_ptrs[row + 1]- 1);
        auto ae = binary_search<int, int>(
            a_T_col_idxs, 855, a_T_row_ptrs[row], a_T_row_ptrs[row + 1] - 1);
    }
    }
}
BENCHMARK(BM_OVERLAP);
*/

// Main for running benchmarks

BENCHMARK_MAIN();

// Main for running tests
/*
int main(int argc, char** argv)
{
    std:: cout<<'\n'<< "--------   Testing   -------- "<<'\n';
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}*/
