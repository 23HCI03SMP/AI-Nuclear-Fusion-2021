#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <omp.h>
#include <string>

#include "traj.h"
//technical parameters
const int n_partd=1E6;
const int n_parte=n_partd;
const int n_output_part=1E10/n_partd; //maximum number of particles to output to file
int n_part[3]= {n_parte,n_partd,n_parte+n_partd}; // 0,number of "super" electrons, electron +deuteriom ions, total
int ndatapoints=2;
int md_me=60;
int nc=1;
int ncalc[2]= {md_me*nc,nc};
int nthreads=8;

int intEon=1;
int intBon=1;
int trig=1;

double r_part_spart=1e7;// ratio of particles per tracked "super" particle

const int n_space=24;// must be even
const int n_space_divx=n_space/2;
const int n_space_divy=n_space/2;
const int n_space_divz=n_space;
int n_space_div[3]= {n_space_divx,n_space_divy,n_space_divz};


int main()
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    omp_set_num_threads(nthreads);

    double t=0;
    auto *pos0=new double[2][n_partd][3];
    auto *pos1=new double[2][n_partd][3];
    auto *posp=new float[2][n_output_part][3];
    // auto *pos1=new double[2][n_partd][3];
    auto *q=new int[2][n_partd]; //+1 or -1
    auto *m=new int[2][n_partd];
    auto *Vfield=new double[n_space_divz][n_space_divy][n_space_divx];
    auto *currentj= new double[2][n_space_divz][n_space_divy][n_space_divx][3];
    auto *Afield= new double[n_space_divz][n_space_divy][n_space_divx][3]; // x,y,z components
    auto *precalc_r=new double[n_space_divz][n_space_divy][n_space_divx];
    auto *np= new int64_t[2][n_space_divz][n_space_divy][n_space_divx];
    auto *It= new int[2][n_space][3];
    auto *KE= new float[2][n_output_part];
    int nt[2]= {0,0};

    ofstream o_file[2];
    ofstream E_file,B_file;

    cout << std::scientific;
    cout.precision(2);
    cerr << std::scientific;
    cerr.precision(2);
    cout<<"float size="<<sizeof(float)<<endl;
    // particle 0 - electron, particle 1 deuteron
    // physical "constantS"

    const double kb=1.38064852e-23; //m^2kss^-2K-1
    //double mp[2]= {9.10938356e-31,3.3435837724e-27}; //kg

    const double e_charge=1.60217662e-19; //C
    const double e_mass=9.10938356e-31;
    const double e_charge_mass=e_charge/e_mass;
    const double kc=8.9875517923e9; //kg m3 s-2 C-2
    const double epsilon0=8.8541878128e-12;//F m-1
    const double pi=3.1415926536;

    //set plasma parameters
    int mp[2] = {1,1835};
    int qs[2]= {-1,1}; // Sign of charge
    double Temp[2]= {1e5,1e5}; // in K convert to eV divide by 1.160451812e4

    //initial bulk electron, ion velocity
    double v0[2][3]= {{0,0,1e7},{0,0,0}};
    // maximum expected magnetic field
    double Bmax=.1;
    // typical dimensions
    double a0=1e-2;
    double poslp[3],poshp[3];
    for (int c=0; c<3; c++)
    {
        poslp[c]=-a0;
        poshp[c]=a0;
    }

    //calculated plasma parameters
    double Density_e=(n_part[1]/ ((poshp[0]-poslp[0])*(poshp[1]-poslp[1])*(poshp[2]-poslp[2])))*r_part_spart;
    double plasma_freq=sqrt(Density_e*e_charge*e_charge_mass/(mp[0]*epsilon0))/(2*pi);
    double plasma_period=1/plasma_freq;
    double Debye_Length=sqrt(epsilon0*kb*Temp[0]/(Density_e*e_charge*e_charge));
    double vel_e=sqrt(kb*Temp[0]/(mp[0]*e_mass));
    double Tv=a0/vel_e/n_space; // time for electron to move across 1 cell
    double Tcyclotron=2.0*pi*mp[0]/(e_charge_mass*Bmax);
    double TDebye=Debye_Length/vel_e;
    //set time step to allow electrons to gyrate if there is B field or to allow electrons to move slowly throughout the plasma distance
    double dt[2];
    // double mu0e_4pi=1e-7*e_charge;
    double Vconst=kc*e_charge*r_part_spart;
    double Aconst=1e-7*e_charge*r_part_spart;
    dt[0]=min(min(TDebye,min(Tv,Tcyclotron)),plasma_period)/md_me; //
    dt[1]=dt[0]*md_me;
    //  double mu0_4pidt[2]= {mu0_4pi/dt[0],mu0_4pi/dt[1]};


    double Ext_E[3],Ext_B[3];
    double dd[3];

    // set initial positions and velocity
    long seed;
    gsl_rng *rng;  // random number generator
    rng = gsl_rng_alloc (gsl_rng_rand48);     // pick random number generator
    seed = time (NULL) * getpid();
    gsl_rng_set (rng, seed);                  // set seed

    for(int p=0; p<2; p++)
    {
        double sigma=sqrt(kb*Temp[p]/(mp[p]*e_mass));
        #pragma omp parallel for reduction(+: nt)
        for(int n=0; n<n_partd; n++)
        {
            //   t[n]=0;
            for(int c=0; c<3; c++)
            {
                pos0[p][n][c]=gsl_ran_flat(rng,poslp[c],poshp[c]);
                pos1[p][n][c]=pos0[p][n][c]+(gsl_ran_gaussian(rng,sigma)+v0[p][c])*dt[p];
            }
            q[p][n]=qs[p];
            m[p][n]=mp[p];
            nt[p]+=q[p][n];
        }
    }
    //get limits and spacing of Field cells

    double posl[3],posh[3];
    for (int c=0; c<3; c++)
    {
        posl[c]=-a0*4*n_space_div[c]/n_space;
        posh[c]=a0*4*n_space_div[c]/n_space;
        dd[c]=(posh[c]-posl[c])/(n_space_div[c]-1);
    }
    //print initial conditions
    {
        std::cout << "electron Temp = " <<Temp[0]<< " K, electron Density = "<< Density_e<<" m^-3" << endl;
        std::cout << "Plasma Frequency(assume cold) = " <<plasma_freq<< " Hz, Plasma period = "<< plasma_period<<" s" << endl;
        std::cout << "Cyclotron period = "<<Tcyclotron<<" s, Time for electron to move across 1 cell = "<<Tv<<" s" << endl;
        std::cout << "electron thermal velocity = "<<vel_e<<endl;
        std::cout << "dt = "<<dt[0]<<" s,"<<endl;
        std::cout << "Debye Length = " <<Debye_Length<< " m, initial dimension = "<< a0<<" m" << endl;
        std::cout << "number of particle per cell = " <<n_partd/(n_space*n_space*n_space)<< endl;

        E_file.open ("info.csv");
//   E_file << std::scientific;
//   E_file.precision(2);
        E_file <<",X, Y, Z"<<endl;
        E_file << "Data Origin," <<posl[0]<<","<<posl[1]<<","<<posl[0]<<endl;
        E_file << "Data Spacing," <<dd[0]<<","<<dd[1]<<","<<dd[2]<<endl;
        E_file << "Data extent x, 0," <<n_space-1<<endl;
        E_file << "Data extent y, 0," <<n_space-1<<endl;
        E_file << "Data extent z, 0," <<n_space-1<<endl;
        E_file << "time step =," <<dt[0]*ncalc[0]<<",s"<<endl;
        E_file << "electron Temp =," <<Temp[0]<< ",K"<<endl;
        E_file << "electron Density =,"<< Density_e<<",m^-3" << endl;
        E_file << "electron thermal velocity = ,"<<vel_e<<endl;
        E_file << "Maximum expected B = ,"<<Bmax<<endl;
        E_file << "Plasma Frequency(assume cold) = " <<plasma_freq<< ", Hz"<<endl;
        E_file << "Plasma period =,"<< plasma_period<<",s" << endl;
        E_file << "Cyclotron period =,"<<Tcyclotron<<",s"<<endl;
        E_file << "Time for electron to move across 1 cell =,"<<Tv<<",s" << endl;
        E_file << "dt = "<<dt[0]<<" s,"<<endl;
        E_file << "Debye Length =," <<Debye_Length<< ",m"<<endl;
        E_file << "initial dimension =,"<< a0<<",m" << endl;
        E_file << "number of particles per cell = " <<n_partd/(n_space*n_space*n_space)<< endl;
        E_file.close();
    }
    for (int ii=0; ii<n_space_divx; ii++)
    {
        double rx2= ii*dd[0];
        rx2*=rx2;
        for (int jj=0; jj<n_space_divy; jj++)
        {
            double ry2= jj*dd[1];
            ry2*=ry2;
            for (int kk=0; kk<n_space_divz; kk++)
            {
                double rz2= kk*dd[2];
                rz2*=rz2;
                if (((ii==0)&(jj==0)&(jj==0)))
                {
                    // assume cells only effect other cells
                    precalc_r[kk][jj][ii]=0;
                }
                else
                {
                    precalc_r[kk][jj][ii]=1/sqrt(rx2+ry2+rz2);
                }
            }
        }
    }
//    save_vti("precalc", 0,n_space_div, posl, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],1,t,(reinterpret_cast<const char*>(precalc_r)),"Float64");
    int i;
    for ( i=0; i<ndatapoints; i++)
    {
        std::cerr << i<<": time = " <<t <<" s" <<", ne = "<<nt[0]<<", ni = "<<nt[1];
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cerr << " Time difference = " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count())/1000 << "[s]" << std::endl;

        Ext_E[0]=0;
        Ext_E[1]=0;
        Ext_E[2]=0;
        Ext_B[0]=0;
        Ext_B[1]=0;
        Ext_B[2]=0;

        //set fields=0 in preparation
        for (int i=0; i<n_space_divx; i++)
        {
            for(int c=0; c<3; c++)
            {
                It[0][i][c]=0;
                It[1][i][c]=0;
            }
            for (int j=0; j<n_space_divy; j++)
            {
                #pragma omp parallel for
                for (int k=0; k<n_space_divz; k++)
                {
                    Vfield[k][j][i]=0;
                    for (int c=0; c<3; c++) Afield[k][j][i][c]=0;
                    for(int p=0; p<2; p++)
                    {
                        np[p][k][j][i]=0;
                        for(int c; c<3; c++) currentj[p][k][j][i][c]=0;

                    }
                }
            }
        }
        nt[0]=0;
        nt[1]=0;
        // find number of particle fields
        for (int p=0; p<2; p++)
        {
            #pragma omp parallel for reduction(+: np[p],nt[p],currentj[p],It[p])
            for(int n=0; n<n_part[p]; n++)
            {
                int i=(int)floor((pos1[p][n][0]-posl[0])/dd[0]+.5);
                int j=(int)floor((pos1[p][n][1]-posl[1])/dd[1]+.5);
                int k=(int)floor((pos1[p][n][2]-posl[2])/dd[2]+.5);
                if ((i>=0)&(i<n_space_divx)&(j>=0)&(j<n_space_divy)&(k>=0)&(k<n_space_divz))
                {
                    np[p][k][j][i]+=q[p][n]; //number of charge (in units of 1.6e-19 C  in each cell
                    nt[p]+=q[p][n];
                    //current density p=0 electron j=nev in each cell n in units 1.6e-19 C m/s
                    for(int c=0; c<3; c++)
                    {
                        currentj[p][k][j][i][c]+=q[p][n]*(pos1[p][n][c]-pos0[p][n][c])/dt[p];
                        It[p][i][c]+=q[p][n]*(pos1[p][n][c]-pos0[p][n][c])/dt[p];
                    }
                }
                else //if out of cells stop the particles from moving
                {
                    q[p][n]=0;
                    for(int c=0; c<3; c++) pos1[p][n][c]=pos0[p][n][c];
                }
            }
        }
// find current
        for(int c=0; c<3; c++)
        {
            for (int i=0; i<n_space_div[c]; i++)
            {
                //           cout <<It[0][i][c]*e_charge*r_part_spart/dd[c]<<",";
            }
            //     cout <<endl;
        }
        //find E field must work out every i,j,k depends on charge in every other cell
        for (int i=0; i<n_space_divx; i++)
        {
            for (int j=0; j<n_space_divy; j++)
            {
                for (int k=0; k<n_space_divz; k++)
                {
                    #pragma omp parallel for reduction(+: Vfield[k][j][i],Afield[k][j][i])
                    for (int ii=0; ii<n_space_divx; ii++)
                    {
                        int iii=abs(ii-i);
                        for (int jj=0; jj<n_space_divy; jj++)
                        {
                            int jjj=abs(jj-j);
                            for (int kk=0; kk<n_space_divz; kk++)
                            {
                                int kkk=abs(kk-k);
                                Vfield[k][j][i]+=(np[1][kk][jj][ii]+np[0][kk][jj][ii])*precalc_r[kkk][jjj][iii];
                                for (int c=0; c<3; c++) Afield[k][j][i][c] +=(currentj[1][kk][jj][ii][c]+currentj[0][kk][jj][ii][c])*precalc_r[kkk][jjj][iii];
                            }
                        }
                    }
                    Vfield[k][j][i]*=Vconst;
                    for (int c=0; c<3; c++) Afield[k][j][i][c]*=Aconst;
                }
            }
        }
        // print out internal Electric potential
        //const char *filename="V";
        save_vti("V", i,n_space_div, posl, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],1,t,(reinterpret_cast<const char*>(Vfield)),"Float64");
        // print out internal electron number
        save_vti("Ne", i,n_space_div, posl, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],1,t,(reinterpret_cast<const char*>(np[0])),"Int64");
        // print out internal magnetic potential file
        save_vti("A", i,n_space_div, posl, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(Afield)),"Float64");
        // print out internal electron current density file
        save_vti("je", i,n_space_div, posl, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(currentj[0])),"Float64");


        //print out some particle positions
        {
            for (int p=0; p<2; p++)
            {
                int nprt=0;
                for(int n=0; n<n_partd; n+=floor(n_partd/n_output_part))
                {
                    double dpos2=0;
                    for (int c=0; c<3; c++)
                    {
                        double dpos=(pos1[p][n][c]-pos0[p][n][c]);
                        dpos*=dpos;
                        dpos2+=dpos;
                    }
                    KE[p][nprt]=0.5*m[p][n]*(dpos2)/(e_charge_mass*dt[p]*dt[p]);
                    //in units of eV
                    for (int c=0; c<3; c++) posp[p][nprt][c]=pos0[0][n][c];
                    nprt++;
                }
            }

            save_vtp("e", i,  n_output_part,1, t, (reinterpret_cast<const char*>(&KE[0][0])), (reinterpret_cast<const char*>(&posp[0][0][0])));
            save_vtp("d", i,  n_output_part,1, t, (reinterpret_cast<const char*>(&KE[1][0])), (reinterpret_cast<const char*>(&posp[1][0][0])));
        }
        t+=dt[0]*ncalc[0];
        //work out motion
        for (int p=0; p<2; p++)
        {
            #pragma omp parallel for
            for(int n=0; n<n_part[p]; n++)
            {
                double qdt_m,qdt2_2m;
                qdt_m=(double)q[p][n]*e_charge_mass*dt[p]/(double)m[p][n];
                qdt2_2m=qdt_m*dt[p];
                //         if (n%10000==0) cout<<qdt_m<<","<<qdt2_2m<<endl;
                for (int jj=0; jj<ncalc[p]; jj++)
                {
                    double dx,dy,dz,r_determinant,px,py,pz,pxx,pxy,pxz,pyy,pzz,pyz;
                    double E[3],B[3];
                    int i=(int)floor((pos1[p][n][0]-posl[0])/dd[0]+.5);
                    int j=(int)floor((pos1[p][n][1]-posl[1])/dd[1]+.5);
                    int k=(int)floor((pos1[p][n][2]-posl[2])/dd[2]+.5);
                    for (int c=0; c<3; c++)
                    {
                        E[c]=Ext_E[c];
                        B[c]=Ext_B[c];
                    }

                    if ((i>0)&(i<n_space_div[0]-1)&(j>0)&(j<n_space_div[1]-1)&(k>0)&(k<n_space_div[2]-1))
                    {
                        //E and B fields right at the surfaces are not calculated and therefore wrong.
                        if(intEon)
                        {
                            E[0]-=(Vfield[k][j][i+1]-Vfield[k][j][i-1])/(2*dd[0]);
                            E[1]-=(Vfield[k][j+1][i]-Vfield[k][j-1][i])/(2*dd[1]);
                            E[2]-=(Vfield[k+1][j][i]-Vfield[k-1][j][i])/(2*dd[2]);
                        }
                        if (intBon)
                        {
                            B[0]+=(Afield[k][j+1][i][2]-Afield[k][j-1][i][2])/(2*dd[1]) - (Afield[k+1][j][i][1]-Afield[k-1][j][i][1])/(2*dd[2]);
                            B[1]+=(Afield[k+1][j][i][0]-Afield[k-1][j][i][0])/(2*dd[2]) - (Afield[k][j][i+1][2]-Afield[k][j][i-1][2])/(2*dd[0]);
                            B[2]+=(Afield[k][j][i+1][1]-Afield[k][j][i-1][1])/(2*dd[0]) - (Afield[k][j+1][i][0]-Afield[k][j-1][i][0])/(2*dd[1]);
                            if (((fabs(B[0])>Bmax)|(fabs(B[1])>Bmax)|(fabs(B[2])>Bmax))&trig)
                            {
                                // If magnetic field is too big, then we have used the wrong time step and need to repeat with smaller time step, by setting Bmax higher.
                                cerr<<"error B>Bmax, jj="<<jj<<", Bx="<<B[0]<<",By="<<B[1]<<",Bz="<<B[2]<<endl;
                                trig=0;
                            }
                        }
                    }
                    px=qdt_m*B[0];
                    py=qdt_m*B[1];
                    pz=qdt_m*B[2];
                    pxx=px*px;
                    pxy=px*py;
                    pxz=px*pz;
                    pyz=py*pz;
                    pyy=py*py;
                    pzz=pz*pz;
                    dx=2*pos1[p][n][0]-pos0[p][n][0]-pz*pos0[p][n][1]+py*pos0[p][n][2];
                    dy=2*pos1[p][n][1]-pos0[p][n][1]-px*pos0[p][n][2]+pz*pos0[p][n][0];
                    dz=2*pos1[p][n][2]-pos0[p][n][2]-py*pos0[p][n][0]+px*pos0[p][n][1];
                    for (int c=0; c<3; c++) pos0[p][n][c]=pos1[p][n][c];

                    r_determinant = 1.0/(1.0+pxx+pyy+pzz);
                    pos1[p][n][0]=r_determinant*((1.0 +pxx)*dx+(pz  +pxy)*dy+(pxz -py )*dz)+qdt2_2m*E[0];
                    pos1[p][n][1]=r_determinant*((pxy -pz )*dx+(1.0 +pyy)*dy+(pyz +px )*dz)+qdt2_2m*E[1];
                    pos1[p][n][2]=r_determinant*((pxz +py )*dx+(pyz -px )*dy+(1.0 +pzz)*dz)+qdt2_2m*E[2];
                    //stop particle
                    if(1)
                    {
                        for (int c=0; c<3; c++)
                        {
                            if(pos1[p][n][c]>posh[c])
                            {
                                pos1[p][n][c]=posh[c];
                                pos0[p][n][c]=posh[c];
                                q[p][n]=0;
                            }
                            if(pos1[p][n][c]<posl[c])
                            {
                                pos1[p][n][c]=posl[c];
                                pos0[p][n][c]=posl[c];
                                q[p][n]=0;
                            }
                        }
                    }
                }
            }
        }
    }

    gsl_rng_free (rng);                       // dealloc the rng


// print out one last time
    std::cerr << i<<": time = " <<t <<" s" <<", ne = "<<nt[0]<<", ni = "<<nt[1];
    //set fields=0 in preparation

    for (int i=0; i<n_space_divx; i++)
    {
        for(int c=0; c<3; c++)
        {
            It[0][i][c]=0;
            It[1][i][c]=0;
        }
        for (int j=0; j<n_space_divy; j++)
        {
            #pragma omp parallel for
            for (int k=0; k<n_space_divz; k++)
            {
                Vfield[k][j][i]=0;
                for (int c=0; c<3; c++) Afield[k][j][i][c]=0;
                for(int p=0; p<2; p++)
                {
                    np[p][k][j][i]=0;
                    for(int c; c<3; c++) currentj[p][k][j][i][c]=0;

                }
            }
        }
    }
    nt[0]=0;
    nt[1]=0;
    // find number of particle fields
    for (int p=0; p<2; p++)
    {
        #pragma omp parallel for reduction(+: np[p],nt[p],currentj[p],It[p])
        for(int n=0; n<n_part[p]; n++)
        {
            int i=(int)floor((pos1[p][n][0]-posl[0])/dd[0]+.5);
            int j=(int)floor((pos1[p][n][1]-posl[1])/dd[1]+.5);
            int k=(int)floor((pos1[p][n][2]-posl[2])/dd[2]+.5);
            if ((i>=0)&(i<n_space_divx)&(j>=0)&(j<n_space_divy)&(k>=0)&(k<n_space_divz))
            {
                np[p][k][j][i]+=q[p][n]; //number of charge (in units of 1.6e-19 C  in each cell
                nt[p]+=q[p][n];
                //current density p=0 electron j=nev in each cell n in units 1.6e-19 C m/s
                for(int c=0; c<3; c++)
                {
                    currentj[p][k][j][i][c]+=q[p][n]*(pos1[p][n][c]-pos0[p][n][c])/dt[p];
                    It[p][i][c]+=q[p][n]*(pos1[p][n][c]-pos0[p][n][c])/dt[p];
                }
            }
            else //if out of cells stop the particles from moving
            {
                q[p][n]=0;
                for(int c=0; c<3; c++) pos1[p][n][c]=pos0[p][n][c];
            }
        }
    }
// find current
    for(int c=0; c<3; c++)
    {
        for (int i=0; i<n_space_div[c]; i++)
        {
            //           cout <<It[0][i][c]*e_charge*r_part_spart/dd[c]<<",";
        }
        //     cout <<endl;
    }
    //find E field must work out every i,j,k depends on charge in every other cell
    //    int di=n_space_divx;
    for (int i=0; i<n_space_divx; i++)
    {
        for (int j=0; j<n_space_divy; j++)
        {
            for (int k=0; k<n_space_divz; k++)
            {
                #pragma omp parallel for reduction(+: Vfield[k][j][i],Afield[k][j][i])
                for (int ii=0; ii<n_space_divx; ii++)
                {
                    int iii=abs(ii-i);
                    for (int jj=0; jj<n_space_divy; jj++)
                    {
                        int jjj=abs(jj-j);
                        for (int kk=0; kk<n_space_divz; kk++)
                        {
                            int kkk=abs(kk-k);
                            Vfield[k][j][i]+=(np[1][kk][jj][ii]+np[0][kk][jj][ii])*precalc_r[kkk][jjj][iii];
                            for (int c=0; c<3; c++) Afield[k][j][i][c] +=(currentj[1][kk][jj][ii][c]+currentj[0][kk][jj][ii][c])*precalc_r[kkk][jjj][iii];
                        }
                    }
                }
                Vfield[k][j][i]*=Vconst;
                for (int c=0; c<3; c++) Afield[k][j][i][c]*=Aconst;
            }
        }
    }
    // print out internal Electric potential
    //const char *filename="V";
    save_vti("V", i,n_space_div, posl, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],1,t,(reinterpret_cast<const char*>(Vfield)),"Float64");
    // print out internal electron number
    save_vti("Ne", i,n_space_div, posl, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],1,t,(reinterpret_cast<const char*>(np[0])),"Int64");
    // print out internal magnetic potential file
    save_vti("A", i,n_space_div, posl, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(Afield)),"Float64");
    // print out internal electron current density file
    save_vti("je", i,n_space_div, posl, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(currentj[0])),"Float64");


    //print out some particle positions
    {
        for (int p=0; p<2; p++)
        {
            int nprt=0;
            for(int n=0; n<n_partd; n+=floor(n_partd/n_output_part))
            {
                double dpos2=0;
                for (int c=0; c<3; c++)
                {
                    double dpos=(pos1[p][n][c]-pos0[p][n][c]);
                    dpos*=dpos;
                    dpos2+=dpos;
                }
                KE[p][nprt]=0.5*m[p][n]*(dpos2)/(e_charge_mass*dt[p]*dt[p]);
                //in units of eV
                for (int c=0; c<3; c++) posp[p][nprt][c]=pos0[p][n][c];
                nprt++;
            }
        }

        save_vtp("e", i,  n_output_part,1, t, (reinterpret_cast<const char*>(&KE[0][0])), (reinterpret_cast<const char*>(&posp[0][0][0])));
        save_vtp("d", i,  n_output_part,1, t, (reinterpret_cast<const char*>(&KE[1][0])), (reinterpret_cast<const char*>(&posp[1][0][0])));
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cerr << "Time difference = " <<(float)( std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count())/1000 << "[s]" << std::endl;
    return 0;
}
