#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <mutex>
#include <vector>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <chrono>


/**
 * @brief CSV File Entry Struct
 */
typedef struct EdgeCSV {
    int id;
    int src;
    int dst;
    double time;
    int int_roll;
    int ext_roll;
} CsvData;

struct T_CSL{
    int indices_;
    double ts_;
    int eid_;

    T_CSL(): indices_(), ts_(), eid_(){};  
    T_CSL(int x, double y, int z): indices_(x), ts_(y), eid_(z){};
};

