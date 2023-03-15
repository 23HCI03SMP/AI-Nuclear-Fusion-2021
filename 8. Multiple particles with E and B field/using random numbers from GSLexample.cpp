#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
using namespace std;
int npart=10000;
int ndatapoints=100;
int ncalc=100;

int main()
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //declare and allocate memory for parameters of each particle
    double *t=new double[npart];
    double *x=new double[npart];
    double *y=new double[npart];
    double *z=new double[npart];
    double *vx=new double[npart];
    double *vy=new double[npart];
    double *vz=new double[npart];
    double *q=new double[npart];
    double *m=new double[npart];
    double *KE=new double[npart];

    long seed;
    gsl_rng *rng;  // random number generator
    rng = gsl_rng_alloc (gsl_rng_rand48);     // pick random number generator
    seed = time (NULL) * getpid();
    gsl_rng_set (rng, seed);                  // set seed
    double x0=0.01; //initial limits of particle position
    double y0=0.01;
    double z0=0.01;
    double sigma=1e6; // standard deviation of x,y,z component of particle velocity.

    ofstream o_file;
    o_file.open ("out.csv");
    o_file << std::scientific;
    o_file.precision(2);
    o_file <<"time, X, Y, Z, Vx, Vy, Vz, KE, q, m, n"<<endl;

    for(int n=0; n<npart; n++)
    {
        t[n]=0;
        x[n]=gsl_ran_flat(rng,-x0,x0);
        y[n]=gsl_ran_flat(rng,-y0,y0);
        z[n]=gsl_ran_flat(rng,-z0,z0);
        vx[n]=gsl_ran_gaussian(rng,sigma);
        vy[n]=gsl_ran_gaussian(rng,sigma);
        vz[n]=gsl_ran_gaussian(rng,sigma);
        q[n]=1.6e-19;
        m[n]=9.11e-31;
    }

    for (int i=0; i<ndatapoints; i++)
    {
        for(int n=0; n<npart; n++)
        {
            KE[n]=0.5*m[n]*sqrt(vx[n]*vx[n]+vy[n]*vy[n]+vz[n]*vz[n]);
            o_file << t[n] <<", ";
            o_file << x[n]  <<", " <<y[n] <<", " <<z[n] <<", ";
            o_file << vx[n] <<", " <<vy[n]<<", " <<vz[n]<<", " <<KE[n]<<", ";
            o_file << q[n]  <<", " <<m[n] <<", " <<   n <<endl;
            for (int j=0; j<ncalc; j++)
            {
                double Ex,Ey,Ez,Bx,By,Bz,dt;
                double dvx,dvy,dvz;
                Ex=0;
                Ey=0;
                Ez=0;
                Bx=0;
                By=0;
                Bz=0.01;
                dt=1e-11;
                t[n]=t[n]+dt;
                //calculate change in velocity
                dvx=q[n]/m[n]*dt*(Ex+vy[n]*Bz-vz[n]*By);
                dvy=q[n]/m[n]*dt*(Ey-vx[n]*Bz+vz[n]*Bx);
                dvz=q[n]/m[n]*dt*(Ez+vx[n]*By-vy[n]*Bx);
                //calculate new velocity
                vx[n]=vx[n]+dvx;
                vy[n]=vy[n]+dvy;
                vz[n]=vz[n]+dvz;
                //calculate new position
                x[n]=x[n]+vx[n]*dt;
                y[n]=y[n]+vy[n]*dt;
                z[n]=z[n]+vz[n]*dt;
            }
        }
    }
    gsl_rng_free (rng);                       // dealloc the rng
    o_file.close();
    delete[] t;
    delete[] x;
    delete[] y;
    delete[] z;
    delete[] vx;
    delete[] vy;
    delete[] vz;
    delete[] q;
    delete[] m;
    delete[] KE;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cerr << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    return 0;
}
