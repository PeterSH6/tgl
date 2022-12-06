#include "gen_graph_multithread.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// #pragma GCC optimize(2)

namespace py = pybind11;
using namespace std;
using namespace chrono;

const int THREAD_NUM = 128;
const int MAX_NODE_NUM = 90000;

// TODO: parameterized this param
const bool add_reverse = false;

std::mutex mtx[MAX_NODE_NUM];
std::vector<CsvData> df; // df



// output
// train set
std::vector<int> int_train_indptr;
std::vector<std::vector<int>> int_train_indices; // std::vector<int> int_train_indices_chain;
std::vector<std::vector<double>> int_train_ts;      // std::vector<int> int_train_ts_chain;
std::vector<std::vector<int>> int_train_eid;     // std::vector<int> int_train_eid_chain;

std::vector<T_CSL> int_train_chain;

// full set
std::vector<int> int_full_indptr;
std::vector<std::vector<int>> int_full_indices; // std::vector<int> int_full_indices_chain;
std::vector<std::vector<double>> int_full_ts;      // std::vector<int> int_full_ts_chain;
std::vector<std::vector<int>> int_full_eid;     // std::vector<int> int_full_eid_chain;

std::vector<T_CSL> int_full_chain;

// ext set
std::vector<int> ext_full_indptr;
std::vector<std::vector<int>> ext_full_indices; // std::vector<int> ext_full_indices_chain;
std::vector<std::vector<double>> ext_full_ts;      // std::vector<int> ext_full_ts_chain;
std::vector<std::vector<int>> ext_full_eid;     // std::vector<int> ext_full_eid_chain;

std::vector<T_CSL> ext_full_chain;

// max_node_id
int max_node_id = -1;

/**
 * @brief Get the chrono duration object
 *
 * @param lo
 * @param hi
 * @return double double value of time in SECONDs
 */
double get_chrono_duration(system_clock::time_point lo, system_clock:: time_point hi) {

    auto duration = duration_cast<microseconds> (hi - lo);

    double time_secis = double(duration.count()) * microseconds::period::num / microseconds::period::den;

    return time_secis;
}

/**
 * @brief linspace as python
 *
 * @tparam T typename
 * @param a lowerbound
 * @param b upperbound
 * @param N slice num
 * @return std::vector<T> sliced vector
 */
template <typename T>
std::vector<T> linspace(T a, T b, size_t N) {
    T h = (b - a) / static_cast<T>(N-1);
    std::vector<T> xs(N);
    typename std::vector<T>::iterator x;
    T val;
    for (x = xs.begin(), val = a; x != xs.end(); ++x, val += h)
        *x = val;

    if(xs[xs.size() - 1] < b) {
        xs[xs.size() - 1] = b;
    }
    return xs;
}

/**
 * @brief itertools.chain as python
 * @param bef array to be chained
 * @param aft array after being chained
 */
void chain_simple(std::vector<std::vector<int>> &bef_idx,
                    std::vector<std::vector<double>> &bef_ts,
                    std::vector<std::vector<int>> &bef_eid,
                    std::vector<T_CSL> &aft){

    // TODO: CHECK_SIZE

    for(int i = 0; i < bef_idx.size(); i++) {
        std::vector<int> each_bef_idx = bef_idx[i];
        std::vector<double> each_bef_ts = bef_ts[i];
        std::vector<int> each_bef_eid = bef_eid[i];

        for(int j = 0; j < each_bef_idx.size(); j++) {

            // printf("going iteration 1th: %d (total: %d); 2th: %d (total: %d)\n",
            //     i, bef_idx.size(), j, each_bef_idx.size());

            // T_CSL csl_data = T_CSL(each_bef_idx[j], each_bef_ts[j], each_bef_eid[j]);
            // aft.push_back(csl_data);
            aft.emplace_back(each_bef_idx[j], each_bef_ts[j], each_bef_eid[j]);
        }
    }

    return ;
}

/**
 * @brief struct sort judging algo (increasing order)
 */
bool cmpar(T_CSL a, T_CSL b) {
    return a.ts_ < b.ts_ ;
}

/**
 * @brief sort partially
 */
void tsort(int idptr_idx, vector<int> &indptr, vector<T_CSL> &arr){
    int beg = indptr[idptr_idx];
    int ed = indptr[idptr_idx + 1];

    // printf("idx: %d, beg: %d , end: %d , arr_size: %d \n", idptr_idx, beg, ed, arr.size());

    std::sort(arr.begin() + beg, arr.begin() + ed, cmpar);

    return ;
}

/**
 * @brief initialize the output of the system
 * including train full ext vectpr
 * follow the T-CSR storage protocol
 * @param num_nodes
 */
void initialize(int num_nodes) {

    for(int i = 0; i < num_nodes + 1; i++) {
        int_train_indptr.push_back(0);
        int_full_indptr.push_back(0);
        ext_full_indptr.push_back(0);
    }

    for(int i = 0; i < num_nodes; i++) {
        std::vector<int> a[10];

        std::vector<double> t[3];

        int_train_indices.push_back(a[0]);
        int_train_ts.push_back(t[0]);
        int_train_eid.push_back(a[2]);

        int_full_indices.push_back(a[3]);
        int_full_ts.push_back(t[1]);
        int_full_eid.push_back(a[5]);

        ext_full_indices.push_back(a[6]);
        ext_full_ts.push_back(t[2]);
        ext_full_eid.push_back(a[8]);
    }
}

/**
 * @brief update T-CSR storage with lock
 */
void update_CSR_with_lock(int lock_id,
                            std::vector<int> &vec_indices,
                            std::vector<double> &vec_ts,
                            std::vector<int> &vec_eid,
                            int dst, double ts, int idx) {
    // lock
    std::lock_guard<std::mutex> lock(mtx[lock_id]);

    vec_indices.emplace_back(dst);
    vec_ts.emplace_back(ts);
    vec_eid.emplace_back(idx);

    return ;
}

/**
 * @brief core function! Build the graph concurrently
 *
 * @param lo thread's responsible's index lowerbound included
 * @param hi thread's responsible's index upperbound excluded
 * @param tid thread's id
 * @param start_ start time of this build
 */
void build_graph(int lo, int hi, int tid, system_clock::time_point very_start_){

    auto start_build_time_ = system_clock::now();

    for(int i = lo; i < hi; i++) {
        CsvData row = df[i];
        int src = row.src;
        int dst = row.dst;
        int int_roll = row.int_roll;

        if(int_roll == 0) {
            update_CSR_with_lock(3 * src + 1,
                int_train_indices[src],
                int_train_ts[src],
                int_train_eid[src],
                dst, row.time, row.id);
            if(add_reverse) {
                update_CSR_with_lock(src,
                    int_train_indices[dst],
                    int_train_ts[dst],
                    int_train_eid[dst],
                    src, row.time, row.id);
            }
        }

        if(int_roll != 3) {
            update_CSR_with_lock(3 * src + 2,
                int_full_indices[src],
                int_full_ts[src],
                int_full_eid[src],
                dst, row.time, row.id);
            if(add_reverse) {
                update_CSR_with_lock(dst,
                    int_full_indices[dst],
                    int_full_ts[dst],
                    int_full_eid[dst],
                    src, row.time, row.id);
            }
        }

        update_CSR_with_lock(3 * src + 3,
            ext_full_indices[src],
            ext_full_ts[src],
            ext_full_eid[src],
            dst, row.time, row.id);
        if(add_reverse) {
            update_CSR_with_lock(dst,
                ext_full_indices[dst],
                ext_full_ts[dst],
                ext_full_eid[dst],
                src, row.time, row.id);
        }


    }

    auto finish_ = system_clock::now();

    // printf("tid: %d responsible range is [%d, %d).\n", tid, lo, hi);
    // printf("tid: %d start the mission in %lf time\n", tid, get_chrono_duration(very_start_, start_build_time_));
    // printf("tid: %d ned the mission in %lf time\n", tid, get_chrono_duration(very_start_, finish_));
    // printf("tid: %d spend %lf time in construction\n", tid, get_chrono_duration(start_build_time_, finish_));
    // printf("\n");

    return ;
}

// read the python data into the df
void read_py(
    std::vector<int> &py_eid,
    std::vector<int> &py_src,
    std::vector<int> &py_dst,
    std::vector<double> &py_ts,
    std::vector<int> &py_int_roll,
    std::vector<int> &py_ext_roll,
    int py_num_nodes,
    int py_num_edges
){
    for(int i = 0; i < py_num_edges; i++) {
        CsvData dt;
        dt.id = py_eid[i];
        dt.src = py_src[i];
        dt.dst = py_dst[i];
        dt.time = py_ts[i];
        dt.int_roll = py_int_roll[i];
        dt.ext_roll = py_ext_roll[i];

        df.push_back(dt);
    }

}

void print_ultimate_result() {

    std::ofstream outfile;
    outfile.open("ultimate_result_cpp.txt");

    outfile<<"Print indptr..." << std::endl;
    for(int i = 0; i < ext_full_indptr.size(); i++) {
        outfile<<ext_full_indptr[i] << ",";
        if(i % 20 == 0) {
            outfile<< std::endl;
        }
    }
    outfile<< std::endl;

    outfile<<"Print indices..." << std::endl;
    for(int i = 0; i < ext_full_chain.size(); i++) {
        outfile<<ext_full_chain[i].indices_ << ",";
        if(i % 20 == 0) {
            outfile<< std::endl;
        }
    }
    outfile<< std::endl;

    outfile<<"Print timestamp..." << std::endl;
    for(int i = 0; i < ext_full_chain.size(); i++) {
        outfile<<ext_full_chain[i].ts_ << ",";
        if(i % 20 == 0) {
            outfile<< std::endl;
        }
    }
    outfile<< std::endl;

    outfile<<"Print eid..." << std::endl;
    for(int i = 0; i < ext_full_chain.size(); i++) {
        outfile<<ext_full_chain[i].eid_ << ",";
        if(i % 20 == 0) {
            outfile<< std::endl;
        }
    }
    outfile<< std::endl;

    outfile.close();
    return ;
}

/**
 * @brief executing function
 */
std::vector<std::vector<double>> run(
    std::vector<int> py_eid,
    std::vector<int> py_src,
    std::vector<int> py_dst,
    std::vector<double> py_ts,
    std::vector<int> py_int_roll,
    std::vector<int> py_ext_roll,
    int py_num_nodes,
    int py_num_edges,
    int py_num_threads,
    std::string py_dataset_name
) {

    auto begin_ = system_clock::now();

    // read the input data
    read_py(py_eid, py_src, py_dst, py_ts, py_int_roll, py_ext_roll, py_num_nodes, py_num_edges);

	auto read_ = system_clock::now();

    int num_nodes = py_num_nodes;
    std::cout<< "the node number is: " << num_nodes << " ." << std::endl;
    std::cout<< "the iteration number is: " << df.size() << " ." << std::endl;

    if(num_nodes == -1 || num_nodes == 0) {
        // TODO: raise error
    }

    // init
    initialize(num_nodes);

    std::thread t[THREAD_NUM];

    // for thread tid: the responsible region is [tid], [tid + 1])
    std::vector<int> thread_range = linspace(0, int(df.size()), THREAD_NUM + 1);

    for (int tid = 0; tid < THREAD_NUM; tid++) {
        t[tid] = std::thread(build_graph, thread_range[tid], thread_range[tid + 1], tid, begin_);
    }

    // sync
    for (int tid = 0; tid < THREAD_NUM; tid++) {
        t[tid].join();
    }

    auto build_end_ = system_clock::now();

    std::cout<< "Build Graph Time by CPP multithread: " << get_chrono_duration(begin_, build_end_) << " seconds." << std::endl;


    // full logic start

    auto logic_start_ = system_clock::now();

    for(int i = 0; i < num_nodes; i++) {
        int_train_indptr[i + 1] = int_train_indptr[i] + int_train_indices[i].size();
        int_full_indptr[i + 1] = int_full_indptr[i] + int_full_indices[i].size();
        ext_full_indptr[i + 1] = ext_full_indptr[i] + ext_full_indices[i].size();
    }

    printf("Chaining...\n");
    // chain
    chain_simple(int_train_indices, int_train_ts, int_train_eid, int_train_chain);
    chain_simple(int_full_indices, int_full_ts, int_full_eid, int_full_chain);
    chain_simple(ext_full_indices, ext_full_ts, ext_full_eid, ext_full_chain);

    printf("Sorting...\n");

    // sort by ts_
    for(int i = 0; i < int_train_indptr.size() - 1; i++) {
        tsort(i, int_train_indptr, int_train_chain);
        tsort(i, int_full_indptr, int_full_chain);
        tsort(i, ext_full_indptr, ext_full_chain);
    }

    auto logic_finish_time_ = system_clock::now();

    std::cout<< "TOTAL Graph Time by CPP multithread: " << get_chrono_duration(begin_, logic_finish_time_) << " seconds." << std::endl;

    print_ultimate_result();

    std::vector<std::vector<double>> ret_result;
    // 0 1 2 3: train indptr, indices, ts, eid
    std::vector<double> abc[12];

    for(int i = 0; i < int_train_indptr.size(); i++){
        abc[0].push_back(int_train_indptr[i]);
    }

    for(int i = 0; i < int_train_chain.size(); i ++) {
        abc[1].push_back(int_train_chain[i].indices_);
        abc[2].push_back(int_train_chain[i].ts_);
        abc[3].push_back(int_train_chain[i].eid_);
    }


    // 4 5 6 7: int_full

    for(int i = 0; i < int_full_indptr.size(); i++){
        abc[4].push_back(int_train_indptr[i]);
    }

    for(int i = 0; i < int_full_chain.size(); i ++) {
        abc[5].push_back(int_full_chain[i].indices_);
        abc[6].push_back(int_full_chain[i].ts_);
        abc[7].push_back(int_full_chain[i].eid_);
    }

    // 8 9 10 11: ext_full

    for(int i = 0; i < ext_full_indptr.size(); i++){
        abc[8].push_back(ext_full_indptr[i]);
    }

    for(int i = 0; i < ext_full_chain.size(); i ++) {
        abc[9].push_back(ext_full_chain[i].indices_);
        abc[10].push_back(ext_full_chain[i].ts_);
        abc[11].push_back(ext_full_chain[i].eid_);
    }

    for(int i = 0; i < 12; i++) {
        ret_result.push_back(abc[i]);
    }


    return ret_result;
}

PYBIND11_MODULE(gen_graph_multithread, m)
{
    // optional module docstring
    m.doc() = "pybind11 example plugin";
    // expose add function, and add keyword arguments and default arguments
    m.def("run", &run, "run the construction function");

}

// int main() {

//     run();

//     // numpy save: start

//     return 0;
// }
