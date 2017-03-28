/*
 *  tsne.cpp
 *  Implementation of both standard and Barnes-Hut-SNE.
 *
 *  Created by Laurens van der Maaten.
 *  Copyright 2012, Delft University of Technology. All rights reserved.
 *
 *  Multicore version by Dmitry Ulyanov, 2016. dmitry.ulyanov.msu@gmail.com
 */

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <glog/logging.h>

#include "quadtree.h"
#include "vptree.h"
#include "tsne.h"


static const int QT_NO_DIMS = 2;

// Perform t-SNE
// X -- double matrix of size [N, D]
// D -- input dimentionality
// Y -- array to fill with the result of size [N, no_dims]
// no_dims -- target dimentionality
void TSNE::run(double* X, double* Y, int _num_threads, int max_iter) {
  
    // Step 1
    step1(X,Y,num_threads,max_iter);
  
    // Step 2
    LOG(INFO) << "Learning embedding";

    int stop_lying_iter = 250, mom_switch_iter = 250;
    double momentum = .5, final_momentum = .8;
    double eta = 200.0;
        
    // Perform main training loop
    int test_iter = 50;
    time_t start = time(0);
    for (int iter = 0; iter < max_iter; iter++) {
      double loss = 0.0;
      step2_one_iter(Y,iter,loss,test_iter);
    }
    time_t end = time(0); _total_time += (float) (end - start) ;

    //LOG(INFO) << "Fitting performed in " << total_time << " seconds";
    LOG(INFO) << "Fitting complete";
}

// step 1
void TSNE::step1(double* X, double* Y, int _num_threads, int max_iter)
{
    if (_N - 1 < 3 * _perplexity) {
      LOG(ERROR) << "Perplexity too large for the number of data points!";
      throw TSNEException("Perplexity too large for the number of data points!");
    }

    num_threads = _num_threads;
    omp_set_num_threads(num_threads);

    LOG(INFO) << "Using no_dims = " << _no_dims << " / perplexity = " << _perplexity << " / theta = " << _theta;

    // Set learning parameters
    float total_time = .0;
    time_t start, end;
    int stop_lying_iter = 250, mom_switch_iter = 250;
    double momentum = .5, final_momentum = .8;
    double eta = 200.0;

    // Allocate some memory
    _dY    = new double[_N*_no_dims];
    _uY    = new double[_N*_no_dims];
    _gains = new double[_N*_no_dims];
    if (_dY == NULL || _uY == NULL || _gains == NULL) { LOG(ERROR) << "Memory allocation failed!"; throw TSNEException("Memory allocation failed"); }
    for (int i = 0; i < _N * _no_dims; i++) {
      _dY[i] = 0.0;
      _uY[i] = 0.0;
      _gains[i] = 1.0;
    }

    // Normalize input data (to prevent numerical problems)
    LOG(INFO) << "Computing input similarities...";
    start = time(0);
    zeroMean(X, _N, _D);
    double max_X = .0;
    for (int i = 0; i < _N * _D; i++) {
        if (X[i] > max_X) max_X = X[i];
    }
    for (int i = 0; i < _N * _D; i++) {
        X[i] /= max_X;
    }

    // Compute input similarities
    int K = static_cast<int>(3*_perplexity);
    _row_P = new int[_N+1];
    _col_P = new int[_N*K];
    _val_P = new double[_N*K];
    if (_row_P == NULL || _col_P == NULL || _val_P == NULL) { LOG(ERROR) << "Memory allocation failed!"; throw TSNEException("Memory allocation failed"); }
    for (int i=0;i<_N*K;i++)
      {
	_col_P[i] = 0;
	_val_P[i] = 0.0;
      }
    
    // Compute asymmetric pairwise input similarities
    computeGaussianPerplexity(X, _N, _D, _row_P, _col_P, _val_P, _perplexity, K);

    // Symmetrize input similarities
    int*    sym_row_P;
    int*    sym_col_P;
    double* sym_val_P;
    symmetrizeMatrix(_row_P, _col_P, _val_P, _N, sym_row_P, sym_col_P, sym_val_P);

    // Return symmetrized matrices
    delete[] _row_P; _row_P = sym_row_P;
    delete[] _col_P; _col_P = sym_col_P;
    delete[] _val_P; _val_P = sym_val_P;
      
    double sum_P = .0;
    for (int i = 0; i < _row_P[_N]; i++) {
        sum_P += _val_P[i];
    }
    for (int i = 0; i < _row_P[_N]; i++) {
        _val_P[i] /= sum_P;
    }

    end = time(0);
    LOG(INFO) << "Done in " << static_cast<float>(end-start) << " seconds (sparsity = " << static_cast<double>(_row_P[_N] / (static_cast<double>(_N) * static_cast<double>(_N))) << " )";

    // Lie about the P-values
    for (int i = 0; i < _row_P[_N]; i++) {
      _val_P[i] *= 12.0;
    }

    // Initialize solution (randomly)
    for (int i = 0; i < _N * _no_dims; i++) {
      Y[i] = randn() * .0001;
    }
}

// step 2
void TSNE::step2_one_iter(double *Y, int &iter, double &loss, const int &test_iter)
{
  // Compute approximate gradient
  computeGradient(_row_P, _col_P, _val_P, Y, _N, _no_dims, _dY, _theta);
  
  
  for (int i = 0; i < _N * _no_dims; i++) {
    // Update gains
    _gains[i] = (sign(_dY[i]) != sign(_uY[i])) ? (_gains[i] + .2) : (_gains[i] * .8);
    if (_gains[i] < .01) {
      _gains[i] = .01;
    }
    
    // Perform gradient update (with momentum and gains)
    _uY[i] = _momentum * _uY[i] - _eta * _gains[i] * _dY[i];
    Y[i] = Y[i] + _uY[i];
  }
  
  // Make solution zero-mean
  zeroMean(Y, _N, _no_dims);
  
  // Stop lying about the P-values after a while, and switch momentum
  if (iter == _stop_lying_iter) {
    for (int i = 0; i < _row_P[_N]; i++) {
      _val_P[i] /= 12.0;
    }
  }
  if (iter == _mom_switch_iter) {
    _momentum = _final_momentum;
  }
  
  // Print out progress
  if ((iter > 0 && iter % test_iter == 0)) {// || (iter == max_iter - 1)) {
    //time_t end = time(0);
    double C = .0;
    
    C = evaluateError(_row_P, _col_P, _val_P, Y, _N, _theta);  // doing approximate computation here!
    
    if (iter == 0)
      LOG(INFO) << "Iteration " << iter + 1 << ": error is " << C;
    else {
      //_total_time += static_cast<float>(end - start);
      LOG(INFO) << "Iteration " << iter << ": error is " << C << std::endl;//" (50 iterations in " << static_cast<float>(end-start) << " seconds)";
    }
    loss = C;
    //start = time(0);
  }
}

// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void TSNE::computeGradient(int* inp_row_P, int* inp_col_P, double* inp_val_P, double* Y, int N, int D, double* dC, double theta)
{

    // Construct quadtree on current map
    QuadTree* tree = new QuadTree(Y, N);

    // Compute all terms required for t-SNE gradient
    double sum_Q = .0;
    double* pos_f = new double[N*D];
    double* neg_f = new double[N*D];
    for (int i=0;i<N*D;i++)
      {
	pos_f[i] = 0.0;
	neg_f[i] = 0.0;
      }
    if (pos_f == NULL || neg_f == NULL) { LOG(ERROR) << "Memory allocation failed!"; throw TSNEException("Memory allocation failed"); }
    tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f);


    #pragma omp parallel for reduction(+:sum_Q)
    for (int n = 0; n < N; n++) {
        double buff[QT_NO_DIMS];
        double this_Q = .0;
        tree->computeNonEdgeForces(n, theta, neg_f + n * D, &this_Q, &buff[0]);
        sum_Q += this_Q;
    }

    // Compute final t-SNE gradient
    for (int i = 0; i < N * D; i++) {
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
    }
    delete[] pos_f;
    delete[] neg_f;
    delete tree;
}

// Evaluate t-SNE cost function (approximately)
double TSNE::evaluateError(int* row_P, int* col_P, double* val_P, double* Y, int N, double theta)
{

    // Get estimate of normalization term
    //const int QT_NO_DIMS = 2;
    QuadTree* tree = new QuadTree(Y, N);
    double buff[QT_NO_DIMS] = {.0, .0};
    double sum_Q = .0;
    for (int n = 0; n < N; n++) {
        double buff1[QT_NO_DIMS];
        tree->computeNonEdgeForces(n, theta, buff, &sum_Q, &buff1[0]);
    }

    // Loop over all edges to compute t-SNE error
    int ind1, ind2;
    double C = .0, Q;
    for (int n = 0; n < N; n++) {
        ind1 = n * QT_NO_DIMS;
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {
            Q = .0;
            ind2 = col_P[i] * QT_NO_DIMS;
            for (int d = 0; d < QT_NO_DIMS; d++) buff[d]  = Y[ind1 + d];
            for (int d = 0; d < QT_NO_DIMS; d++) buff[d] -= Y[ind2 + d];
            for (int d = 0; d < QT_NO_DIMS; d++) Q += buff[d] * buff[d];
            Q = (1.0 / (1.0 + Q)) / sum_Q;
            C += val_P[i] * log((val_P[i] + FLT_MIN) / (Q + FLT_MIN));
        }
    }
    return C;
}

// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
void TSNE::computeGaussianPerplexity(double* X, int N, int D, int* _row_P, int* _col_P, double* _val_P, double perplexity, int K) {
  if (perplexity > K) LOG(ERROR) << "Perplexity should be lower than K!";

    int* row_P = _row_P;
    int* col_P = _col_P;
    double* val_P = _val_P;

    row_P[0] = 0;
    for (int n = 0; n < N; n++) {
        row_P[n + 1] = row_P[n] + K;
    }

    // Build ball tree on data set
    VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
    std::vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    for (int n = 0; n < N; n++) {
        obj_X[n] = DataPoint(D, n, X + n * D);
    }
    tree->create(obj_X);

    // Loop over all points to find nearest neighbors
    LOG(INFO) << "Building tree...";

    int steps_completed = 0;
    #pragma omp parallel for
    for (int n = 0; n < N; n++)
    {
        std::vector<double> cur_P(K);
        std::vector<DataPoint> indices;
        std::vector<double> distances;

        // Find nearest neighbors
        tree->search(obj_X[n], K + 1, &indices, &distances);

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -DBL_MAX;
        double max_beta =  DBL_MAX;
        double tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0; double sum_P;
        while (!found && iter < 200) {

            // Compute Gaussian kernel row
            for (int m = 0; m < K; m++) {
                cur_P[m] = exp(-beta * distances[m + 1]);
            }

            // Compute entropy of current row
            sum_P = DBL_MIN;
            for (int m = 0; m < K; m++) {
                sum_P += cur_P[m];
            }
            double H = .0;
            for (int m = 0; m < K; m++) {
                H += beta * (distances[m + 1] * cur_P[m]);
            }
            H = (H / sum_P) + log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - log(_perplexity);
            if (Hdiff < tol && -Hdiff < tol) {
                found = true;
            }
            else {
                if (Hdiff > 0) {
                    min_beta = beta;
                    if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else {
                    max_beta = beta;
                    if (min_beta == -DBL_MAX || min_beta == DBL_MAX)
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for (int m = 0; m < K; m++) {
            cur_P[m] /= sum_P;
        }
        for (int m = 0; m < K; m++) {
            col_P[row_P[n] + m] = indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }

        // Print progress
        #pragma omp atomic
        ++steps_completed;

        if (steps_completed % 10000 == 0)
        {
            #pragma omp critical
	  LOG(INFO) << " - point " << steps_completed << " of " << N;
        }
    }

    // Clean up memory
    obj_X.clear();
    delete tree;
}


void TSNE::symmetrizeMatrix(int* _row_P, int* _col_P, double* _val_P, int N, int*& sym_row_P, int*& sym_col_P, double*& sym_val_P) {

    // Get sparse matrix
    int* row_P = _row_P;
    int* col_P = _col_P;
    double* val_P = _val_P;

    // Count number of elements and row counts of symmetric matrix
    int* row_counts = new int[N];
    for (int i=0;i<N;i++)
      row_counts[i] = 0;
    if (row_counts == NULL) { LOG(ERROR) << "Memory allocation failed!"; throw TSNEException("Memory allocation failed"); }
    for (int n = 0; n < N; n++) {
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if (col_P[m] == n) present = true;
            }
            if (present) row_counts[n]++;
            else {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for (int n = 0; n < N; n++) no_elem += row_counts[n];

    // Allocate memory for symmetrized matrix
    sym_row_P = new int[N+1];
    sym_col_P = new int[no_elem];
    sym_val_P = new double[no_elem];
    if (sym_row_P == NULL || sym_col_P == NULL || sym_val_P == NULL) { LOG(ERROR) << "Memory allocation failed!"; throw TSNEException("Memory allocation failed"); }

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for (int n = 0; n < N; n++) sym_row_P[n + 1] = sym_row_P[n] + row_counts[n];

    // Fill the result matrix
    int* offset = new int[N];
    for (int i=0;i<N;i++)
      offset[i] = 0;
    if (offset == NULL) { LOG(ERROR) << "Memory allocation failed!"; throw TSNEException("Memory allocation failed"); }
    for (int n = 0; n < N; n++) {
        for (int i = row_P[n]; i < row_P[n + 1]; i++) {                                 // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++) {
                if (col_P[m] == n) {
                    present = true;
                    if (n <= col_P[i]) {                                                // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if (!present) {
                sym_col_P[sym_row_P[n]        + offset[n]]        = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n]        + offset[n]]        = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if (!present || (present && n <= col_P[i])) {
                offset[n]++;
                if (col_P[i] != n) offset[col_P[i]]++;
            }
        }
    }

    // Divide the result by two
    for (int i = 0; i < no_elem; i++) sym_val_P[i] /= 2.0;

    // Free up some memory
    delete[] offset; offset = nullptr;
    delete[] row_counts; row_counts  = nullptr;
}


// Makes data zero-mean
void TSNE::zeroMean(double* X, int N, int D) {

    // Compute data mean
    double* mean = new double[D];
    for (int i=0;i<D;i++)
      mean[i] = 0.0;
    if (mean == NULL) { LOG(ERROR) << "Memory allocation failed!"; throw TSNEException("Memory allocation failed"); }
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            mean[d] += X[n * D + d];
        }
    }
    for (int d = 0; d < D; d++) {
        mean[d] /= (double) N;
    }

    // Subtract data mean
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            X[n * D + d] -= mean[d];
        }
    }
    delete[] mean; mean = nullptr;
}


// Generates a Gaussian random number
double TSNE::randn() {
    double x, y, radius;
    do {
        x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
        y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
        radius = (x * x) + (y * y);
    } while ((radius >= 1.0) || (radius == 0.0));
    radius = sqrt(-2 * log(radius) / radius);
    x *= radius;
    y *= radius;
    return x;
}

// XXX: for Python
extern "C"
{
    extern void tsne_run_double(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, int _num_threads, int max_iter)
    {
        printf("Performing t-SNE using %d cores.\n", _num_threads);
        TSNE tsne(N,D,perplexity,theta);
        tsne.run(X, Y, _num_threads, max_iter);
    }
}
