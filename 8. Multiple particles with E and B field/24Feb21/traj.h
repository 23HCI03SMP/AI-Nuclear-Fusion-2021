#ifndef TRAJ_H_INCLUDED
#define TRAJ_H_INCLUDED

using namespace std;
void save_vti(string filename, int i,int n_space_div[3], double posl[3], double dd[3], uint64_t num,int ncomponents, double t, const char* data,string typeofdata);
void save_pvd(string filename, int ndatapoints);
void save_vtp(string filename, int i,  uint64_t num,int ncomponents, double t, const char* data, const char* points);
#endif // TRAJ_H_INCLUDED
