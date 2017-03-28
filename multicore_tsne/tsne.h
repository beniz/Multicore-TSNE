/*
 *  tsne.h
 *  Header file for t-SNE.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 *  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
 */


#ifndef TSNE_H
#define TSNE_H

#include <exception>

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }


class TSNEException : public std::exception
{
 public:
  TSNEException(const std::string &s)
    :_s(s) {}
  ~TSNEException() {}
  const char* what() const noexcept { return _s.c_str(); }
 private:
  std::string _s;
};

class TSNE
{
public:
   TSNE() {}
   TSNE(const int &N, const int &D, const int &perplexity=30, const double &theta=0.5)
     :_N(N),_D(D),_theta(theta),_perplexity(perplexity) {}
  ~TSNE()
    {
      delete[] _row_P;
      delete[] _col_P;
      delete[] _val_P;
      delete[] _Y;
      delete[] _dY;
      delete[] _uY;
      delete[] _gains;
    }
  
    void run(double* X, double* Y, int num_threads, int max_iter);
    void step1(double* X, double* Y, int num_threads, int max_iter);
    void step2_one_iter(double *Y, int &iter, double &loss, const int &test_iter);
    
    void symmetrizeMatrix(int* row_P, int* col_P, double* val_P, int N, int*& sym_row_P, int*& sym_col_P, double*& sym_val_P);
private:
    int num_threads;
    void computeGradient(int* inp_row_P, int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta);
    double evaluateError(int* row_P, int* col_P, double* val_P, double* Y, int N, double theta);
    void zeroMean(double* X, int N, int D);
    void computeGaussianPerplexity(double* X, int N, int D, int* _row_P, int* _col_P, double* _val_P, double perplexity, int K);
    double randn();

 public:
    int *_row_P = nullptr;
    int *_col_P = nullptr;
    double *_val_P = nullptr;
    double *_Y = nullptr;
    double *_dY = nullptr;
    double *_uY = nullptr;
    double *_gains = nullptr;
    int _N;
    int _D;
    const int _no_dims = 2;
    double _theta = 0.5;
    int _perplexity = 30;
    int _iter = 0;
    double _momentum = 0.5;
    const double _final_momentum = 0.8;
    double _eta = 200.0;
    int _stop_lying_iter = 250;
    int _mom_switch_iter = 250;
    float _total_time = 0.0;
};

#endif

