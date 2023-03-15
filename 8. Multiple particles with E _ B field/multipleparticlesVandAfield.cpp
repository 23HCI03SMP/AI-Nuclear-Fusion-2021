#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <omp.h>
#include <string>

using namespace std;

//technical parameters
int n_partd=5E5;
int n_parte=n_partd;
int n_output_part=5000; //maximum number of particles to output to file
int n_part[3]= {0,n_parte,n_parte+n_partd}; // 0,number of "super" electrons, electron +deuteriom ions, total
int ndatapoints=10;
int md_me=60;
int nc=1;
int ncalc[2]= {md_me*nc,nc};
int nthreads=8;

int intEon=0;
int intBon=0;
int trig=1;

double r_part_spart=1e2;// ratio of particles per tracked "super" particle

const int n_space=20;
const int n_space_divx=n_space;
const int n_space_divy=n_space;
const int n_space_divz=n_space;

int main()
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    omp_set_num_threads(nthreads);

//    double *t=new double[n_part[2]];
    double t=0;
    double *x0=new double[n_part[2]];
    double *y0=new double[n_part[2]];
    double *z0=new double[n_part[2]];
    double *x1=new double[n_part[2]];
    double *y1=new double[n_part[2]];
    double *z1=new double[n_part[2]];
    int *q=new int[n_part[2]]; //+1 or -1
    int *m=new int[n_part[2]];
    auto *Vfield=new double[n_space_divx][n_space_divy][n_space_divz];
    auto *jx= new double[2][n_space_divx][n_space_divy][n_space_divz];
    auto *jy= new double[2][n_space_divx][n_space_divy][n_space_divz];
    auto *jz= new double[2][n_space_divx][n_space_divy][n_space_divz];
    auto *Afieldx= new double[n_space_divx][n_space_divy][n_space_divz];
    auto *Afieldy= new double[n_space_divx][n_space_divy][n_space_divz];
    auto *Afieldz= new double[n_space_divx][n_space_divy][n_space_divz];
    auto *np= new int[2][n_space_divx][n_space_divy][n_space_divz];
    auto *Itx= new int[2][n_space_divx];
    auto *Ity= new int[2][n_space_divy];
    auto *Itz= new int[2][n_space_divz];
    int nt[2]= {0,0};

    ofstream o_file[2];
    ofstream E_file,B_file;
    o_file[0].open ("e.csv");
    o_file[0] << std::scientific;
    o_file[0].precision(2);
    o_file[1].open ("d.csv");
    o_file[1] << std::scientific;
    o_file[1].precision(2);
    o_file[0] <<"time, X, Y, Z, n, KE"<<endl;
    o_file[1] <<"time, X, Y, Z, n, KE"<<endl;



    cout << std::scientific;
    cout.precision(2);

    // particle 0 - electron, particle 1 deuteron
    // physical "constantS"

    double kb=1.38064852e-23; //m^2kss^-2K-1
    //double mp[2]= {9.10938356e-31,3.3435837724e-27}; //kg

    double e_charge=1.60217662e-19; //C
    double e_mass=9.10938356e-31;
    double e_charge_mass=e_charge/e_mass;
    double kc=8.9875517923e9; //kg m3 s-2 C-2
    double epsilon0=8.8541878128e-12;//F m-1
    double pi=3.1415926536;

    //set plasma parameters
    int mp[2] = {1,1835};
    int qs[2]= {-1,1}; // Sign of charge
    double Temp[2]= {1e1,1e1}; // in K convert to eV divide by 1.160451812e4
    double Bmax=10; // maximum expected magnetic field
    double a0=1e-2;
    double xlp=-a0;
    double ylp=-a0;
    double zlp=-a0;
    double xhp=a0;
    double yhp=a0;
    double zhp=a0;
    //calculated plasma parameters
    double Density_e=(n_part[1]/ ((xhp-xlp)*(yhp-ylp)*(zhp-zlp)))*r_part_spart;
    double plasma_freq=sqrt(Density_e*e_charge*e_charge_mass/(mp[0]*epsilon0))/(2*pi);
    double plasma_period=1/plasma_freq;
    double Debye_Length=sqrt(epsilon0*kb*Temp[0]/(Density_e*e_charge*e_charge));
    double vel_e=sqrt(kb*Temp[0]/(mp[0]*e_mass));
    double Tv=a0/vel_e/n_space_divx; // time for electron to move across 1 cell
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

    E_file.open ("fields.csv");
//   E_file << std::scientific;
//   E_file.precision(2);
    std::cout << "electron Temp = " <<Temp[0]<< " K, electron Density = "<< Density_e<<" m^-3" << endl;
    std::cout << "Plasma Frequency(assume cold) = " <<plasma_freq<< " Hz, Plasma period = "<< plasma_period<<" s" << endl;
    std::cout << "Cyclotron period = "<<Tcyclotron<<" s, Time for electron to move across 1 cell = "<<Tv<<" s" << endl;
    std::cout << "dt = "<<dt[0]<<" s,"<<endl;
    std::cout << "Debye Length = " <<Debye_Length<< " m, initial dimension = "<< a0<<" m" << endl;
    std::cout << "number of particle per cell = " <<n_partd/(n_space_divx*n_space_divy*n_space_divz)<< endl;
    double Ext_Ex,Ext_Ey,Ext_Ez,Ext_Bx,Ext_By,Ext_Bz;
    double ddx,ddy,ddz;


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
        for(int n=n_part[p]; n<n_part[p+1]; n++)
        {
         //   t[n]=0;
            x0[n]=gsl_ran_flat(rng,xlp,xhp);
            y0[n]=gsl_ran_flat(rng,ylp,yhp);
            z0[n]=gsl_ran_flat(rng,zlp,zhp);

            x1[n]=x0[n]+gsl_ran_gaussian(rng,sigma)*dt[p];
            y1[n]=y0[n]+gsl_ran_gaussian(rng,sigma)*dt[p];
            z1[n]=z0[n]+gsl_ran_gaussian(rng,sigma)*dt[p];
            q[n]=qs[p];
            m[n]=mp[p];
            nt[p]+=q[n];
        }
    }
    //get limits and spacing of Field cells

    double xh=a0*2;
    double xl=-a0*2;
    double yh=a0*2;
    double yl=-a0*2;
    double zh=a0*2;
    double zl=-a0*2;
    ddx=(xh-xl)/(n_space_divx-1);
    ddy=(yh-yl)/(n_space_divy-1);
    ddz=(zh-zl)/(n_space_divz-1);

    E_file <<",X, Y, Z"<<endl;
    E_file << "Data Origin," <<xl<<","<<yl<<","<<zl<<endl;
    E_file << "Data Spacing," <<ddx<<","<<ddy<<","<<ddz<<endl;
    E_file << "Scalar Array Name, V"<<endl;
    E_file << "Data extent x, 0," <<n_space_divx-1<<endl;
    E_file << "Data extent y, 0," <<n_space_divy-1<<endl;
    E_file << "Data extent z, 0," <<n_space_divz-1<<endl;
    E_file << "time step," <<dt[0]*ncalc[0]<<endl;
    E_file << "electron Temp = " <<Temp[0]<< " K, electron Density = "<< Density_e<<" m^-3" << endl;
    E_file << "Plasma Frequency(assume cold) = " <<plasma_freq<< " Hz, Plasma period = "<< plasma_period<<" s" << endl;
    E_file << "Cyclotron period = "<<Tcyclotron<<" s, Time for electron to move across 1 cell = "<<Tv<<" s" << endl;
    E_file << "dt = "<<dt[0]<<" s,"<<endl;
    E_file<< "Debye Length = " <<Debye_Length<< " m, initial dimension = "<< a0<<" m" << endl;
    E_file << "number of particle per cell = " <<n_partd/(n_space_divx*n_space_divy*n_space_divz)<< endl;
    for (int i=0; i<ndatapoints; i++)
    {
        std::cerr << i<<": time = " <<t <<" s" <<", ne = "<<nt[0]<<", ni = "<<nt[1];
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cerr << " Time difference = " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count())/1000 << "[s]" << std::endl;

        Ext_Ex=0;
        Ext_Ey=0;
        Ext_Ez=0;
        Ext_Bx=0;
        Ext_By=0;
        Ext_Bz=0;

        //set fields=0 in preparation
        //    reinterpret_cast< double*>(Afieldx)
        for (int i=0; i<n_space_divx; i++)
        {
            Itx[0][i]=0;
            Itx[1][i]=0;
            Ity[0][i]=0;
            Ity[1][i]=0;
            Itz[0][i]=0;
            Itz[1][i]=0;
            for (int j=0; j<n_space_divy; j++)
            {
                #pragma omp parallel for
                for (int k=0; k<n_space_divz; k++)
                {
                    Vfield[i][j][k]=0;
                    Afieldx[i][j][k]=0;
                    Afieldy[i][j][k]=0;
                    Afieldz[i][j][k]=0;
                    for(int p=0; p<2; p++)
                    {
                        np[p][i][j][k]=0;
                        jx[p][i][j][k]=0;
                        jy[p][i][j][k]=0;
                        jz[p][i][j][k]=0;
                    }
                }
            }
        }
        nt[0]=0;
        nt[1]=0;
        // find number of particle fields
        for (int p=0; p<2; p++)
        {
            #pragma omp parallel for reduction(+: np[p],nt,jx[p],jy[p],jz[p])
            for(int n=n_part[p]; n<n_part[p+1]; n++)
            {
                int i=(int)floor((x1[n]-xl)/ddx+.5);
                int j=(int)floor((y1[n]-yl)/ddy+.5);
                int k=(int)floor((z1[n]-zl)/ddz+.5);
                if ((i>=0)&(i<n_space_divx)&(j>=0)&(j<n_space_divy)&(k>=0)&(k<n_space_divz))
                {
                    np[p][i][j][k]+=q[n]; //number of charge (in units of 1.6e-19 C  in each cell
                    nt[p]+=q[n];
                    //current density p=0 electron j=nev in each cell n in units 1.6e-19 C m/s
                    jx[p][i][j][k]+=q[n]*(x1[n]-x0[n])/dt[p];
                    jy[p][i][j][k]+=q[n]*(y1[n]-y0[n])/dt[p];
                    jz[p][i][j][k]+=q[n]*(z1[n]-z0[n])/dt[p];
                    Itx[p][i]+=q[n]*(x1[n]-x0[n])/dt[p];
                    Ity[p][j]+=q[n]*(y1[n]-y0[n])/dt[p];
                    Itz[p][k]+=q[n]*(z1[n]-z0[n])/dt[p];
                }
                else //if out of cells stop the particles from moving
                {
                    q[n]=0;
                    x1[n]=x0[n];
                    y1[n]=y0[n];
                    z1[n]=z0[n];
                }
            }
        }
        for (int i=0; i<n_space_divx; i++)
        {
            cout <<Itx[0][i]*e_charge*r_part_spart/ddx<<",";
        }
        cout <<endl;
        //find E field must work out every i,j,k depends on charge in every other cell
        //    int di=n_space_divx;
        for (int i=0; i<n_space_divx; i++)
        {
            double xx=xl+i*ddx;
            for (int j=0; j<n_space_divy; j++)
            {
                double yy=yl+j*ddy;
                for (int k=0; k<n_space_divz; k++)
                {
                    double zz=zl+k*ddz;
                    #pragma omp parallel for reduction(+: Vfield[i][j][k],Afieldx[i][j][k],Afieldy[i][j][k],Afieldz[i][j][k])
                    for (int ii=0; ii<n_space_divx; ii++)
                    {
                        double rx= xx-xl-ii*ddx;
                        for (int jj=0; jj<n_space_divy; jj++)
                        {
                            double ry= yy-yl-jj*ddy;
                            for (int kk=0; kk<n_space_divz; kk++)
                            {
                                double rz= zz-zl-kk*ddz;
                                if (((rx==0)&(ry==0)&(rz==0)))
                                {
                                    //               cout <<"z" <<rx<<","<<ry<<","<<rz<<",   ";
                                }
                                else
                                {
                                    double r_1=1/sqrt(rx*rx+ry*ry+rz*rz);
                                    double dnp=(np[1][ii][jj][kk]+np[0][ii][jj][kk]);
                                    Vfield[i][j][k]+=dnp*r_1;
                                    Afieldx[i][j][k] +=(jx[1][ii][jj][kk]+jx[0][ii][jj][kk])*r_1;
                                    Afieldy[i][j][k] +=(jy[1][ii][jj][kk]+jy[0][ii][jj][kk])*r_1;
                                    Afieldz[i][j][k] +=(jz[1][ii][jj][kk]+jz[0][ii][jj][kk])*r_1;
                                }

                            }
                        }
                    }
                    Vfield[i][j][k]*=Vconst;
                    Afieldx[i][j][k]*=Aconst;
                    Afieldy[i][j][k]*=Aconst;
                    Afieldz[i][j][k]*=Aconst;

                }
            }
        }
        std::ofstream osV("./out/V_"+to_string(i)+".vti", std::ios::binary | std::ios::out);
        std::ofstream osA("./out/A_"+to_string(i)+".raw", std::ios::binary | std::ios::out);
        osV<< "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\"> \n ";
        osV<<"<ImageData WholeExtent=\"0 ";
        osV<<to_string(n_space_divx-1)+" 0" +to_string(n_space_divy-1)+" 0 "+to_string(n_space_divy-1)+"\"";
        osV<<"Origin=\""+to_string(xl)+" "+to_string(yl)+" "+to_string(zl)+"\"";
        osV<<" Spacing=\""+to_string(ddx)+" "+to_string(ddy)+" "+to_string(ddz)+"\"";
        osV<<"Direction=\"1 0 0 0 1 0 0 0 1\"> \n";
        osV<< "<FieldData>\n";
//        osV<<  "<DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"appended\" RangeMin=\"0\"  RangeMax=\"100000\" offset=\"0\"                   />\n";
        osV<<  "<DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"appended\" RangeMin=\"0\" RangeMax=\"0\" offset=\"0\"/>\n";
//        osV<<  "<DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"appended\" offset=\"0\"/>\n";
        osV<<"</FieldData>\n";
        osV<<"<Piece Extent=\"0";
        osV<<to_string(n_space_divx-1)+" 0" +to_string(n_space_divy-1)+" 0 "+to_string(n_space_divy-1)+"\"\n";
        osV<<"<PointData Scalars=\"V\">\n";
//       osV<<"<DataArray type=\"Float64\" Name=\"V\" format=\"appended\" RangeMin=\"-0.018027235883\" RangeMax=\"0.024445251412\"  offset=\"16\" />\n";
        osV<<"<DataArray type=\"Float64\" Name=\"V\" format=\"appended\" RangeMin=\"0\" RangeMax=\"0\" offset=\"16\" />\n";
//       osV<<"<DataArray type=\"Float64\" Name=\"V\" format=\"appended\" offset=\"16\" />\n";

        osV<<"  </PointData>\n";
        osV<<"<CellData>\n";
        osV<<"  </CellData>\n";
        osV<<"</Piece>\n";
        osV<<"</ImageData>\n";
        osV<<"<AppendedData encoding=\"raw\">_";
        uint64_t num=8;
        osV.write(reinterpret_cast<const char*>(&num), std::streamsize(sizeof(num)));
        osV.write(reinterpret_cast<const char*>(&t), std::streamsize(sizeof(double)));
        num=n_space_divx*n_space_divy*n_space_divz*8;
        //     cout <<num<<endl;
        osV.write(reinterpret_cast<const char*>(&num), std::streamsize(sizeof(num)));
        osV.write(reinterpret_cast<const char*>(Vfield), std::streamsize(num*sizeof(double)));
        osV<<"</AppendedData>\n";
        osV<<"</VTKFile>";

        osA.write(reinterpret_cast<const char*>(Afieldx), std::streamsize(n_space_divx*n_space_divy*n_space_divz*sizeof(double)));
        osV.close();
        osA.close();
        for (int p=0; p<2; p++)
        {
            for(int n=n_part[p]; n<n_part[p+1]; n=n+ceil(n_parte/n_output_part))
            {
                double dx2=(x1[n]-x0[n]);
                dx2*=dx2;
                double dy2=(y1[n]-y0[n]);
                dy2*=dy2;
                double dz2=(z1[n]-z0[n]);
                dz2*=dz2;
                double KE=0.5*m[n]*(dx2+dy2+dz2)/(e_charge_mass*dt[p]*dt[p]);
                //in units of eV
                o_file[p]<<t<<","<<x0[n]<<","<<y0[n]<<"," <<z0[n] <<"," <<n <<","<<KE<<endl;
            }
        }
        t+=dt[0]*ncalc[0];
        //work out motion
        for (int p=0; p<2; p++)
        {
            #pragma omp parallel for
            for(int n=n_part[p]; n<n_part[p+1]; n++)
            {
                double qdt_m,qdt2_2m;
                qdt_m=(double)q[n]*e_charge_mass*dt[p]/(double)m[n];
                qdt2_2m=qdt_m*dt[p];

                for (int jj=0; jj<ncalc[p]; jj++)
                {
                    double dx,dy,dz,r_determinant,px,py,pz,pxx,pxy,pxz,pyy,pzz,pyz;
                    double Ex,Ey,Ez,Bx,By,Bz;
                    int i=(int)floor((x1[n]-xl)/ddx+.5);
                    int j=(int)floor((y1[n]-yl)/ddy+.5);
                    int k=(int)floor((z1[n]-zl)/ddz+.5);
                    Ex=Ext_Ex;
                    Ey=Ext_Ey;
                    Ez=Ext_Ez;
                    Bx=Ext_Bx;
                    By=Ext_By;
                    Bz=Ext_Bz;
                    if ((i>0)&(i<n_space_divx-1)&(j>0)&(j<n_space_divy-1)&(k>0)&(k<n_space_divz-1))
                    {
                        if(intEon)
                        {
                            Ex-=(Vfield[i+1][j][k]-Vfield[i-1][j][k])/(2*ddx);
                            Ey-=(Vfield[i][j+1][k]-Vfield[i][j-1][k])/(2*ddy);
                            Ez-=(Vfield[i][j][k+1]-Vfield[i][j][k-1])/(2*ddz);
                        }
                        if (intBon)
                        {
                            Bx+=(Afieldz[i][j+1][k]-Afieldz[i][j-1][k])/(2*ddy) - (Afieldy[i][j][k+1]-Afieldy[i][j][k-1])/(2*ddz);
                            By+=(Afieldx[i][j][k+1]-Afieldx[i][j][k-1])/(2*ddz) - (Afieldz[i+1][j][k]-Afieldz[i-1][j][k])/(2*ddx);
                            Bz+=(Afieldy[i+1][j][k]-Afieldy[i-1][j][k])/(2*ddx) - (Afieldx[i][j+1][k]-Afieldz[i][j-1][k])/(2*ddy);
                            if (((fabs(Bx)>Bmax)|(fabs(By)>Bmax)|(fabs(Bz)>Bmax))&trig)
                            {
                                cerr<<"error B>Bmax, jj="<<jj<<", Bx="<<Bx<<",By="<<By<<",Bz="<<Bz<<endl;
                                trig=0;
                            }
                        }
                    }
                    px=qdt_m*Bx;
                    py=qdt_m*By;
                    pz=qdt_m*Bz;
                    pxx=px*px;
                    pxy=px*py;
                    pxz=px*pz;
                    pyz=py*pz;
                    pyy=py*py;
                    pzz=pz*pz;
                    dx=2*x1[n]-x0[n]-pz*y0[n]+py*z0[n];
                    dy=2*y1[n]-y0[n]-px*z0[n]+pz*x0[n];
                    dz=2*z1[n]-z0[n]-py*x0[n]+px*y0[n];
                    x0[n]=x1[n];
                    y0[n]=y1[n];
                    z0[n]=z1[n];
                    r_determinant = 1.0/(1.0+pxx+pyy+pzz);
                    x1[n]=r_determinant*((1.0 +pxx)*dx+(pz  +pxy)*dy+(pxz -py )*dz)+qdt2_2m*Ex;
                    y1[n]=r_determinant*((pxy -pz )*dx+(1.0 +pyy)*dy+(pyz +px )*dz)+qdt2_2m*Ey;
                    z1[n]=r_determinant*((pxz +py )*dx+(pyz -px )*dy+(1.0 +pzz)*dz)+qdt2_2m*Ez;
                    //stop particle
                    if(x1[n]>xh) x1[n]=xh;
                    if(x1[n]<xl) x1[n]=xl;
                    if(y1[n]>yh) y1[n]=yh;
                    if(y1[n]<yl) y1[n]=yl;
                    if(z1[n]>zh) z1[n]=zh;
                    if(z1[n]<zl) z1[n]=zl;
                }
            }
        }
    }
    gsl_rng_free (rng);                       // dealloc the rng
    o_file[0].close();
    o_file[1].close();
    E_file.close();
    std::ofstream osV("V.pvd", std::ios::binary | std::ios::out);
    osV<<"<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    osV<< "<Collection>\n";
    for (int i=0; i<ndatapoints; i++)
    {
        osV <<"<DataSet timestep=\""+to_string(i)+"\" part=\"0\" file=\"out/V_"+to_string(i)+".vti\"/>\n";
    }
    osV<< " </Collection>\n";
    osV<<"</VTKFile>\n";

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cerr << "Time difference = " <<(float)( std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count())/1000 << "[s]" << std::endl;
    return 0;
}
