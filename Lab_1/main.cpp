#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <immintrin.h>
using std::vector;
using std::cout;
using std::endl;
using namespace std::chrono;

using doubleMtrx = vector<vector<double>>;
using mtrxMtrx = vector<vector<doubleMtrx>>;

//_________________SERVICE__________________________________________________________________
bool cmpTest(const mtrxMtrx &mtrx_1, const mtrxMtrx &mtrx_2) {
    for (size_t i = 0; i < mtrx_1.size(); i++) {
        for (size_t j = 0; j < mtrx_1[0].size(); j++) {
            for (size_t k = 0; k < 8; k++) {
                for (size_t m = 0; m < 8; m++) {
                    double num1 = mtrx_1[i][j][k][m];
                    double num2 = mtrx_2[i][j][k][m];

                    if (abs(num1 - num2) > 0.000000001) {
                        cout << "Compare test: \033[1;31mfailed\033[0m" << endl;
                        return false;
                    }
                }
            }
        }
    }
    cout << "Compare test: \033[1;32msuccess\033[0m" << endl;
    return true;
}
void randMtrx(mtrxMtrx &mtrx) {
    std::random_device device;
    std::mt19937 generator{device()};
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    for (auto &i: mtrx)
        for (auto &j: i)
            for (auto &k: j)
                for (auto &l: k)
                    l = distribution(generator);
}
//________________________________SERVICE_END________________________________________________

//____________________________________AUTO_VECTOR_________________________________________
__attribute__((target("avx")))
doubleMtrx opMltply_dM(const doubleMtrx & lhs, const doubleMtrx & rhs) {
    doubleMtrx res (lhs.size(), vector<double>(rhs[0].size()));

    for (size_t i = 0; i < lhs.size(); i++) {
        for (size_t k = 0; k < lhs[0].size(); k++) {
            for (size_t j = 0; j < rhs[0].size(); j++) {
                res[i][j] += lhs[i][k] * rhs[k][j];
            }
        }
    }

    return res;
}
//__attribute__((optimize("no-tree-vectorize")))
__attribute__((target("avx")))
void opAddAs_dM(doubleMtrx & lhs, const doubleMtrx & rhs) {
    for (size_t i = 0; i < lhs.size(); i++)
        for (size_t j = 0; j < rhs[0].size(); j++)
            lhs[i][j] += rhs[i][j];
}

__attribute__((target("avx")))
mtrxMtrx opMltply_mM(const mtrxMtrx & lhs, const mtrxMtrx & rhs) {
    mtrxMtrx res (lhs.size(), vector<doubleMtrx>(rhs[0].size(),doubleMtrx(8, vector<double>(8,0))));

    for (size_t i = 0; i < lhs.size(); i++)
        for (size_t k = 0; k < lhs[0].size(); k++)  
            for (size_t j = 0; j < rhs[0].size(); j++)
                opAddAs_dM(res[i][j], opMltply_dM(lhs[i][k], rhs[k][j]));

    return res;
}
//____________________________________AUTO_VECTOR_END________________________________________

//___________________________________NO_AUTO_VECTOR_________________________________________
__attribute__((target("no-avx")))
// __attribute__((optimize("no-tree-vectorize")))
doubleMtrx opMltply_dM_noV(const doubleMtrx & lhs, const doubleMtrx & rhs) {
    doubleMtrx res (lhs.size(), vector<double>(rhs[0].size()));

    for (size_t i = 0; i < lhs.size(); i++) {
        for (size_t k = 0; k < lhs[0].size(); k++) {
            for (size_t j = 0; j < rhs[0].size(); j++) {
                res[i][j] += lhs[i][k] * rhs[k][j];
            }
        }
    }

    return res;
}
//__attribute__((optimize("no-tree-vectorize")))
__attribute__((target("no-avx")))
void opAddAs_dM_noV(doubleMtrx & lhs, const doubleMtrx & rhs) {
    for (size_t i = 0; i < lhs.size(); i++)
        for (size_t j = 0; j < rhs[0].size(); j++)
            lhs[i][j] += rhs[i][j];
}

 __attribute__((target("no-avx")))
// __attribute__((optimize("no-tree-vectorize")))
mtrxMtrx opMltply_mM_noV(const mtrxMtrx & lhs, const mtrxMtrx & rhs) {
    mtrxMtrx res (lhs.size(), vector<doubleMtrx>(rhs[0].size(),doubleMtrx(8, vector<double>(8,0))));

    for (size_t i = 0; i < lhs.size(); i++)
        for (size_t k = 0; k < lhs[0].size(); k++)  
            for (size_t j = 0; j < rhs[0].size(); j++)
                opAddAs_dM_noV(res[i][j], opMltply_dM_noV(lhs[i][k], rhs[k][j]));

    return res;
}
//___________________________________NO_AUTO_VECTOR_END________________________________________

//___________________________________MANUAL_VECT_____________________________________________
__attribute__((target("avx")))
// __attribute__((optimize("no-tree-vectorize")))
doubleMtrx opMltply_dM_mnl(const doubleMtrx & lhs, const doubleMtrx & rhs) {
    doubleMtrx res (lhs.size(), vector<double>(rhs[0].size()));

    for (size_t i = 0; i < lhs.size(); i++) {
        auto sum_l = _mm256_set1_pd(0.0);
        auto sum_r = _mm256_set1_pd(0.0);
        for (size_t k = 0; k < lhs[0].size(); k++) {
            auto b = _mm256_set1_pd(lhs[i][k]);
            // for (size_t j = 0; j < rhs[0].size(); j += 4) {
                auto a = _mm256_loadu_pd(&rhs[k][0]);
                auto r = _mm256_mul_pd(a, b);
                sum_l = _mm256_add_pd(sum_l, r);

                a = _mm256_loadu_pd(&rhs[k][4]);
                r = _mm256_mul_pd(a, b);
                sum_r = _mm256_add_pd(sum_r, r);
                // res[i][j] += lhs[i][k] * rhs[k][j];
            // }
        }
        _mm256_storeu_pd(&res[i][0], sum_l);
        _mm256_storeu_pd(&res[i][4], sum_r);
    }

    return res;
}
//__attribute__((optimize("no-tree-vectorize")))
__attribute__((target("avx")))
void opAddAs_dM_mnl(doubleMtrx & lhs, const doubleMtrx & rhs) {
    for (size_t i = 0; i < lhs.size(); i++) {
        for (size_t j = 0; j < rhs[0].size(); j += 4) {
            // lhs[i][j] += rhs[i][j];
            auto sum = _mm256_loadu_pd(&lhs[i][j]);
            auto a = _mm256_loadu_pd(&rhs[i][j]);
            sum = _mm256_add_pd(sum, a);
            _mm256_storeu_pd(&lhs[i][j], sum);
        }
    }
}

 __attribute__((target("avx")))
// __attribute__((optimize("no-tree-vectorize")))
mtrxMtrx opMltply_mM_mnl(const mtrxMtrx & lhs, const mtrxMtrx & rhs) {
    mtrxMtrx res (lhs.size(), vector<doubleMtrx>(rhs[0].size(),doubleMtrx(8, vector<double>(8,0))));

    for (size_t i = 0; i < lhs.size(); i++)
        for (size_t k = 0; k < lhs[0].size(); k++)  
            for (size_t j = 0; j < rhs[0].size(); j++)
                opAddAs_dM_mnl(res[i][j], opMltply_dM_mnl(lhs[i][k], rhs[k][j]));

    return res;
}
//___________________________________MANUAL_VECT_END____________________________________________

int main() {
    const int N = 150;
    const int M = 180;
    const int K = 300;
    // mtrxMtrx A (M, vector<doubleMtrx>(K, doubleMtrx(8, vector<double>(8, 2))));
    // mtrxMtrx B (K, vector<doubleMtrx>(N, doubleMtrx(8, vector<double>(8, 3))));
    mtrxMtrx A (M, vector<doubleMtrx>(K, doubleMtrx(8, vector<double>(8,0))));
    mtrxMtrx B (K, vector<doubleMtrx>(N, doubleMtrx(8, vector<double>(8,0))));
    randMtrx(A);
    randMtrx(B);
    auto start = high_resolution_clock::now();
    mtrxMtrx vecM = opMltply_mM(A, B);
    auto time = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    cout << "Time: auto vectorization: " << time << " ms." << endl;
 
    start = high_resolution_clock::now();
    mtrxMtrx novectM = opMltply_mM_noV(A, B);
    time = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    cout << "Time: no auto vectorization: " << time << " ms." << endl;

    start = high_resolution_clock::now();
    mtrxMtrx mnlvecM = opMltply_mM_mnl(A, B);
    time = duration_cast<milliseconds>(high_resolution_clock::now() - start).count();
    cout << "Time: manual vectorization: " << time << " ms." << endl;

    cmpTest(vecM, mnlvecM);

    return 0;
}
//bd93f9