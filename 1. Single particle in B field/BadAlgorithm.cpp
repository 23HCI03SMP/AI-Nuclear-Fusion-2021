#include <iostream>
#include <chrono>

using namespace std;
int npart=1;
int ndatapoints=100;
int ncalc=10;

int main()
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    double *x=new double[npart];
    double *y=new double[npart];
    double *z=new double[npart];
    double *vx=new double[npart];
    double *vy=new double[npart];
    double *vz=new double[npart];
    double *q=new double[npart];
    double *m=new double[npart];
    std::cout << std::scientific;
    std::cout.precision(2);

    for(int n=0; n<npart; n++)
    {
        x[n]=0;
        y[n]=0;
        z[n]=0;
        vx[n]=1e6;
        vy[n]=2e6;
        vz[n]=3e5;
        q[n]=1.6e-19;
        m[n]=9.11e-31;
    }
    for (int i=0; i<ndatapoints; i++)
    {
        for(int n=0; n<npart-1; n++)
        {
            cout << x[n] <<", "<<y[n]<<", "<<z[n]<<", ";
        }
        cout << x[npart-1] <<", "<<y[npart-1]<<", "<<z[npart-1]<< endl;
        for(int n=0; n<npart; n++)
        {

            for (int j=0; j<ncalc; j++)
            {
                double Ex,Ey,Ez,Bx,By,Bz,dt;
                double dvx,dvy,dvz;
                Ex=0;
                Ey=0;
                Ez=0;
                Bx=0;
                By=0;
                Bz=0;
                dt=1e-9;
                dvx=q[n]/m[n]*dt*(Ex+vy[n]*Bz-vz[n]*By);
                dvy=q[n]/m[n]*dt*(Ey-vx[n]*Bz+vz[n]*Bx);
                dvz=q[n]/m[n]*dt*(Ez+vx[n]*By-vy[n]*Bx);
                vx[n]=vx[n]+dvx;
                vy[n]=vy[n]+dvy;
                vz[n]=vz[n]+dvz;
                x[n]=x[n]+vx[n]*dt;
                y[n]=y[n]+vy[n]*dt;
                z[n]=z[n]+vz[n]*dt;
            }
        }

    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cerr << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;
    return 0;
}
