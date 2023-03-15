#ifndef TRAJ_H_INCLUDED
#define TRAJ_H_INCLUDED
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <omp.h>
#include <string>
#include <CL/cl.hpp>
using namespace std;
void save_vti(string filename, int i,int n_space_div[3], float posl[3], float dd[3], uint64_t num,int ncomponents, double t, const char* data,string typeofdata,int sizeofdata);
void save_pvd(string filename, int ndatapoints);
void save_vtp(string filename, int i,  uint64_t num,int ncomponents, double t, const char* data, const char* points);

void set_initial_pos_vel(int n_part_types,int n_particles, float *pos0, float *pos1,float *sigma, int *q,int *m,int *nt);
#endif // TRAJ_H_INCLUDED
