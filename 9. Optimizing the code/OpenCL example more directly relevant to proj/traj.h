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
#define CL_HPP_TARGET_OPENCL_VERSION 220
#include <CL/cl2.hpp>
using namespace std;
void save_vti(string filename, int i,unsigned int n_space_div[3], float posl[3], float dd[3], uint64_t num,int ncomponents, double t, const char* data,string typeofdata,int sizeofdata);
void save_pvd(string filename, int ndatapoints);
void save_vtp(string filename, int i,  uint64_t num,int ncomponents, double t, const char* data, const char* points);
void set_initial_pos_vel(int n_part_types,int n_particles, float *pos0, float *pos1,float *sigma, int *q,int *m,int *nt);
//void cl_start();
void cl_start(cl::Context &context1,cl::Device &default_device1,cl::Program &program1);
//void mult(float);
void mult(float *C,float *A, float *B,int n, cl::Context &context,cl::Device &default_device,cl::Program &program);
void calc_pnn(float *px,float *py,float *pz,float *qdt2_2mEx,float *qdt2_2mEy,float *qdt2_2mEz,float *pos0x,float *pos0y,float *pos0z,float *pos1x,float *pos1y, float *pos1z,float *qdt_m, int n, cl::Context &context,cl::Device &default_device,cl::Program &program);
//technical parameters
const unsigned int n_space=32;// must be even or 2 to power of n
const int n_partd=n_space*n_space*n_space*256 ;
const int n_parte=n_partd;
const int n_output_part=min(n_partd,8192); //maximum number of particles to output to file
const int nprtd=floor(n_partd/n_output_part);
const int n_part[3]= {n_parte,n_partd,n_parte+n_partd}; // 0,number of "super" electrons, electron +deuteriom ions, total
const int ndatapoints=3;
const int md_me=60;
const int nc=1;
const int ncalc[2]= {md_me*nc,nc};
const int nthreads=8;

const int intEon=1;
const int intBon=1;
const float Bmax=30;
const int trig=1;

const float r_part_spart=1e14/n_partd;// ratio of particles per tracked "super" particle

//const unsigned int n_space=32;// must be 2 to power of n
const unsigned int n_space_divx=n_space;
const unsigned int n_space_divy=n_space;
const unsigned int n_space_divz=n_space;

#endif // TRAJ_H_INCLUDED
