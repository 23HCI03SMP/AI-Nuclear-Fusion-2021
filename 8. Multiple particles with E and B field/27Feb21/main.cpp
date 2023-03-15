#include "traj.h"
//technical parameters
const int n_partd=1e7;
const int n_parte=n_partd;
const int n_output_part=min(n_partd,10000); //maximum number of particles to output to file
int n_part[3]= {n_parte,n_partd,n_parte+n_partd}; // 0,number of "super" electrons, electron +deuteriom ions, total
int ndatapoints=20;
int md_me=60;
int nc=1;
int ncalc[2]= {md_me*nc,nc};
int nthreads=4;

int intEon=1;
int intBon=1;
int trig=1;

float r_part_spart=1e12/n_partd;// ratio of particles per tracked "super" particle

const int n_space=32;// must be even
const int n_space_divx=n_space/2;
const int n_space_divy=n_space/2;
const int n_space_divz=n_space;
int n_space_div[3]= {n_space_divx,n_space_divy,n_space_divz};


int main()
{
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    omp_set_num_threads(nthreads);


    double t=0;
    auto *pos0=new float[2][n_partd][3];
    auto *pos1=new float[2][n_partd][3];
    auto *posp=new float[2][n_output_part][3];
    auto *q=new int[2][n_partd]; //+1 or -1
    auto *m=new int[2][n_partd];
//   auto *Vfield=new float[n_space_divz][n_space_divy][n_space_divx];
    auto *currentj= new float[2][n_space_divz][n_space_divy][n_space_divx][3];
//    auto *Afield= new float[n_space_divz][n_space_divy][n_space_divx][3]; // x,y,z components
    auto *E= new float[n_space_divz][n_space_divy][n_space_divx][3];
    auto *dEdx= new float[n_space_divz][n_space_divy][n_space_divx][3];
    auto *dEdy= new float[n_space_divz][n_space_divy][n_space_divx][3];
    auto *dEdz= new float[n_space_divz][n_space_divy][n_space_divx][3];
    auto *B= new float[n_space_divz][n_space_divy][n_space_divx][3];
    auto *dBdx= new float[n_space_divz][n_space_divy][n_space_divx][3];
    auto *dBdy= new float[n_space_divz][n_space_divy][n_space_divx][3];
    auto *dBdz= new float[n_space_divz][n_space_divy][n_space_divx][3];
    auto *precalc_r=new float[n_space_divz][n_space_divy][n_space_divx];
    auto *precalc_r2=new float[n_space_divz][n_space_divy][n_space_divx];
    auto *precalc_r3=new float[n_space_divz][n_space_divy][n_space_divx];
    auto *np= new int32_t[2][n_space_divz][n_space_divy][n_space_divx];
    auto *It= new int[2][n_space][3];
    auto *KE= new float[2][n_output_part];
    int nt[2]= {0,0};

    ofstream o_file[2];
    ofstream E_file,B_file;

    cout << std::scientific;
    cout.precision(2);
    cerr << std::scientific;
    cerr.precision(2);
    cout<<"float size="<<sizeof(float)<<", ";
    cout<<"int size="<<sizeof(int)<<endl;
    // particle 0 - electron, particle 1 deuteron
    // physical "constantS"

    const float kb=1.38064852e-23; //m^2kss^-2K-1
    //float mp[2]= {9.10938356e-31,3.3435837724e-27}; //kg

    const float e_charge=1.60217662e-19; //C
    const float e_mass=9.10938356e-31;
    const float e_charge_mass=e_charge/e_mass;
    const float kc=8.9875517923e9; //kg m3 s-2 C-2
    const float epsilon0=8.8541878128e-12;//F m-1
    const float pi=3.1415926536;

    //set plasma parameters
    int mp[2] = {1,1835*2};
    int qs[2]= {-1,1}; // Sign of charge
    float Temp[2]= {1e5,1e5}; // in K convert to eV divide by 1.160451812e4

    //initial bulk electron, ion velocity
    float v0[2][3]= {{0,0,1e7},{0,0,0}};
    // maximum expected magnetic field
    float Bmax=1;
    // typical dimensions
    float a0=1e-2;
    float posLp[3],posHp[3];
    for (int c=0; c<3; c++)
    {
        posLp[c]=-a0;
        posHp[c]=a0;
    }

    //calculated plasma parameters
    float Density_e=(n_part[1]/ ((posHp[0]-posLp[0])*(posHp[1]-posLp[1])*(posHp[2]-posLp[2])))*r_part_spart;
    float plasma_freq=sqrt(Density_e*e_charge*e_charge_mass/(mp[0]*epsilon0))/(2*pi);
    float plasma_period=1/plasma_freq;
    float Debye_Length=sqrt(epsilon0*kb*Temp[0]/(Density_e*e_charge*e_charge));
    float vel_e=sqrt(kb*Temp[0]/(mp[0]*e_mass));
    float Tv=a0/vel_e/n_space; // time for electron to move across 1 cell
    float Tcyclotron=2.0*pi*mp[0]/(e_charge_mass*Bmax);
    float TDebye=Debye_Length/vel_e;
    //set time step to allow electrons to gyrate if there is B field or to allow electrons to move slowly throughout the plasma distance
    float dt[2];
    // float mu0e_4pi=1e-7*e_charge;
    float Vconst=kc*e_charge*r_part_spart;
    float Aconst=1e-7*e_charge*r_part_spart;
    dt[0]=min(min(TDebye,min(Tv,Tcyclotron)),plasma_period)/md_me; //
    dt[1]=dt[0]*md_me;
    //  float mu0_4pidt[2]= {mu0_4pi/dt[0],mu0_4pi/dt[1]};


    float Ext_E[3],Ext_B[3];
    float dd[3];

    std::cerr << " Time difference = " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count())/1000 << "[s]" << std::endl;

    // set initial positions and velocity
    float sigma[2]= {sqrt(kb*Temp[0]/(mp[0]*e_mass)),sqrt(kb*Temp[1]/(mp[1]*e_mass))};
    long seed;
    gsl_rng *rng;  // random number generator
    rng = gsl_rng_alloc (gsl_rng_rand48);     // pick random number generator
    seed = time (NULL) * getpid();
    gsl_rng_set (rng, seed);                  // set seed

    for(int p=0; p<2; p++)
    {

        #pragma omp parallel for reduction(+: nt)
        for(int n=0; n<n_partd; n++)
        {
            //   t[n]=0;
            for(int c=0; c<3; c++)
            {
                pos0[p][n][c]=gsl_ran_flat(rng,posLp[c],posHp[c]);
                pos1[p][n][c]=pos0[p][n][c]+(gsl_ran_gaussian(rng,sigma[p])+v0[p][c])*dt[p];
            }
            if (n==0) cout << "p = " <<p <<", sigma = " <<sigma[p]<<", temp = " << Temp[p] << ",mass of particle = " << mp[p] << dt[p]<<endl;
            q[p][n]=qs[p];
            m[p][n]=mp[p];
            nt[p]+=q[p][n];
        }
    }
    gsl_rng_free (rng);                       // dealloc the rng

    cout <<"v0 electron = "<<v0[0][0]<<","<<v0[0][1]<<","<<v0[0][2]<<endl;
//get limits and spacing of Field cells

    std::cerr << " Set initial random positions: Time difference = " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count())/1000 << "[s]" << std::endl;

    float posL[3],posH[3];
    for (int c=0; c<3; c++)
    {
        posL[c]=-a0*4*n_space_div[c]/n_space;
        posH[c]=a0*4*n_space_div[c]/n_space;
        dd[c]=(posH[c]-posL[c])/(n_space_div[c]-1);
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
        E_file << "Data Origin," <<posL[0]<<","<<posL[1]<<","<<posL[0]<<endl;
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
        float rx2= ii*dd[0];
        rx2*=rx2;
        for (int jj=0; jj<n_space_divy; jj++)
        {
            float ry2= jj*dd[1];
            ry2*=ry2;
            for (int kk=0; kk<n_space_divz; kk++)
            {
                float rz2= kk*dd[2];
                rz2*=rz2;
                if (((ii==0)&(jj==0)&(jj==0)))
                {
                    // assume cells only effect other cells
                    precalc_r[kk][jj][ii]=0;
                }
                else
                {
                    precalc_r3[kk][jj][ii]=1/sqrt(rx2+ry2+rz2);
                    precalc_r[kk][jj][ii]=precalc_r3[kk][jj][ii];
                    precalc_r3[kk][jj][ii]*=precalc_r3[kk][jj][ii];
                    precalc_r2[kk][jj][ii]=precalc_r3[kk][jj][ii];
                    precalc_r3[kk][jj][ii]*=precalc_r3[kk][jj][ii];
                }
            }
        }
    }
//    save_vti("precalc", 0,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],1,t,(reinterpret_cast<const char*>(precalc_r)),"Float32");
    int i;
    for ( i=0; i<ndatapoints; i++)
    {
        std::cerr << i<<": time = " <<t <<" s" <<", ne = "<<nt[0]<<", ni = "<<nt[1];
        std::cerr << " Time difference = " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count())/1000 << "[s]" << std::endl;

        Ext_E[0]=0;
        Ext_E[1]=0;
        Ext_E[2]=0;
        Ext_B[0]=0;
        Ext_B[1]=0;
        Ext_B[2]=0;

        //     float *It_lin=);
        for (i=0; i<2*n_space_div[0]*3; i++) (reinterpret_cast< float*>(It))[i]=0;
        cout<<"current"<<endl;
        //set fields=0 in preparation
        #pragma omp parallel for
        for (i=0; i< n_space_div[0]*n_space_div[1]*n_space_div[2]*3; i++)(reinterpret_cast< float*>(B))[i]=0;
        #pragma omp parallel for
        for (i=0; i< n_space_div[0]*n_space_div[1]*n_space_div[2]*3; i++)(reinterpret_cast< float*>(E))[i]=0;
          cout<<"E and B"<<endl;
        #pragma omp parallel for
        for (i=0; i<n_space_div[0]*n_space_div[1]*n_space_div[2]*2; i++) (reinterpret_cast< int32_t*>(np))[i]=0;
        cout<<"np"<<endl;
        #pragma omp parallel for
        for (i=0; i<n_space_div[0]*n_space_div[1]*n_space_div[2]*2*3; i++) (reinterpret_cast< float*>(currentj))[i]=0;
        cout<<"currentdensity"<<endl;
        nt[0]=0;
        nt[1]=0;
        // find number of particle fields
        for (int p=0; p<2; p++)
        {
            #pragma omp parallel for reduction(+: np[p],nt[p],currentj[p],It[p])
            for(int n=0; n<n_partd; n++)
            {
                int i=(int)floor((pos1[p][n][0]-posL[0])/dd[0]+.5);
                int j=(int)floor((pos1[p][n][1]-posL[1])/dd[1]+.5);
                int k=(int)floor((pos1[p][n][2]-posL[2])/dd[2]+.5);
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
            }
        }
// find current
/*
        for(int c=0; c<3; c++)
        {
            for (int i=0; i<n_space_div[c]; i++)
            {
                //           cout <<It[0][i][c]*e_charge*r_part_spart/dd[c]<<",";
            }
            //     cout <<endl;
        }
*/
        //find E field must work out every i,j,k depends on charge in every other cell
        for (int i=0; i<n_space_divx; i++)
        {
            for (int j=0; j<n_space_divy; j++)
            {
                for (int k=0; k<n_space_divz; k++)
                {
                    #pragma omp parallel for reduction(+: E[k][j][i],B[k][j][i])
                    for (int ii=0; ii<n_space_divx; ii++)
                    {
                        float ddd[3];
                        ddd[0]=(ii-i)*dd[0];
                        int iii=abs(ii-i);
                        for (int jj=0; jj<n_space_divy; jj++)
                        {
                            ddd[1]=(jj-j)*dd[1];
                            int jjj=abs(jj-j);

                            for (int kk=0; kk<n_space_divz; kk++)
                            {
                                int kkk=abs(kk-k);
                                ddd[2]=(kk-k)*dd[2];
                                for (int c=0; c<3; c++)
                                {
                                    if (intEon)   E[k][j][i][c] -=(np[1][kk][jj][ii]+np[0][kk][jj][ii])*precalc_r3[kkk][jjj][iii]*ddd[c];
                                    if (intBon)   B[k][j][i][c] +=(currentj[1][kk][jj][ii][c]+currentj[0][kk][jj][ii][c])*precalc_r2[kkk][jjj][iii];
                                }
                            }
                        }
                    }

                    for (int c=0; c<3; c++)
                    {
                        E[k][j][i][c]*=Vconst;
                        B[k][j][i][c]*=Aconst;
                    }
                }
            }
        }

        for (int i=1; i<n_space_divx-1; i++)
        {
            for (int j=1; j<n_space_divy-1; j++)
            {
                #pragma omp parallel for
                for (int k=1; k<n_space_divz-1; k++)
                {
                    //E and B fields right at the surfaces are not calculated and therefore wrong.
                    if(intEon)
                    {
                        for (int c=0; c<3; c++)
                        {
                            dEdx[k][j][i][c]=(E[k][j][i+1][c]-E[k][j][i-1][c])/(2*dd[0]);
                            dEdy[k][j][i][c]=(E[k][j+1][i][c]-E[k][j-1][i][c])/(2*dd[1]);
                            dEdz[k][j][i][c]=(E[k+1][j][i][c]-E[k-1][j][i][c])/(2*dd[2]);
                        }
                    }
                    if (intBon)
                    {
                        for (int c=0; c<3; c++)
                        {
                            dBdx[k][j][i][c]=(B[k][j][i+1][c]-B[k][j][i-1][c])/(2*dd[0]);
                            dBdy[k][j][i][c]=(B[k][j+1][i][c]-B[k][j-1][i][c])/(2*dd[1]);
                            dBdz[k][j][i][c]=(B[k+1][j][i][c]-B[k-1][j][i][c])/(2*dd[2]);
                        }
                    }
                }
            }
        }
        std::cerr << " calc E&B Time difference = " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count())/1000 << "[s]" << std::endl;

        // print out internal Electric potential
//const char *filename="V";
//   save_vti("V", i,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],1,t,(reinterpret_cast<const char*>(Vfield)),"Float32");
        /*        for (int i =0; i<n_space_div[0]; i++)
                {
                    cout <<E[0][0][i][0]<<",";
                }
                cout <<endl;
         */
        save_vti("E", i,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(E)),"Float32",sizeof(float));
// print out internal electron number
        save_vti("Ne", i,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],1,t,(reinterpret_cast<const char*>(np[0])),"Int32",sizeof(int));
// print out internal magnetic potential file
//   save_vti("A", i,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(Afield)),"Float32"),8;
        save_vti("B", i,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(B)),"Float32",sizeof(float));
// print out internal electron current density file
        save_vti("je", i,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(currentj[0])),"Float32",sizeof(float));

        //print out some particle positions
        {
            for (int p=0; p<2; p++)
            {
                int nprt=0;
                for(int n=0; n<n_partd; n+=floor(n_partd/n_output_part))
                {
                    float dpos2=0;
                    for (int c=0; c<3; c++)
                    {
                        float dpos=(pos1[p][n][c]-pos0[p][n][c]);
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
        std::cerr << " print data: Time difference = " << (float)(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count())/1000 << "[s]" << std::endl;

        t+=dt[0]*ncalc[0];
        //work out motion
        for (int p=0; p<2; p++)
        {
            cout<<p<<endl;
            // #pragma omp parallel for
            for(int n=0; n<n_part[p]; n++)
            {
                float qdt_m,qdt2_2m;

                qdt_m=(float)q[p][n]*e_charge_mass*dt[p]/(float)m[p][n];
                qdt2_2m=qdt_m*dt[p];
                //         if (n%10000==0) cout<<qdt_m<<","<<qdt2_2m<<endl;
                for (int jj=0; jj<ncalc[p]; jj++)
                {
                    float dx,dy,dz;
                    float r_determinant,px,py,pz,pxx,pxy,pxz,pyy,pzz,pyz,M11,M12,M13,M21,M22,M23,M31,M32,M33;
                    //assume particles do not move out of cell
                    float dpos1_l[3]= {pos1[p][n][0]-posL[0],pos1[p][n][1]-posL[1],pos1[p][n][2]-posL[2]};
                    //         int i=(int)floor((pos1[p][n][0]-posL[0])/dd[0]+.5);
                    //         int j=(int)floor((pos1[p][n][1]-posL[1])/dd[1]+.5);
                    //         int k=(int)floor((pos1[p][n][2]-posL[2])/dd[2]+.5);
                    unsigned int i=min((int)floor(dpos1_l[0]/dd[0]+.5),n_space_div[0]-1);
                    unsigned int j=min((int)floor(dpos1_l[1]/dd[1]+.5),n_space_div[1]-1);
                    unsigned int k=min((int)floor(dpos1_l[2]/dd[2]+.5),n_space_div[2]-1);
                    //                 cout<<i<<","<<j<<","<<k<<endl;
                    float dpos1_c[3]= {dpos1_l[0]-i*dd[0],dpos1_l[1]-j*dd[1],dpos1_l[2]-k*dd[2]};
                    float qdt2_2mE[3]= { qdt2_2m*(E[k][j][i][0]+ dEdx[k][j][i][0]*dpos1_c[0]+ dEdy[k][j][i][0]*dpos1_c[1]+dEdz[k][j][i][0]*dpos1_c[2])+Ext_E[0],\
                                         qdt2_2m*(E[k][j][i][1]+ dEdx[k][j][i][1]*dpos1_c[0]+ dEdy[k][j][i][1]*dpos1_c[1]+dEdz[k][j][i][1]*dpos1_c[2])+Ext_E[1],\
                                         qdt2_2m*(E[k][j][i][2]+ dEdx[k][j][i][2]*dpos1_c[0]+ dEdy[k][j][i][2]*dpos1_c[1]+dEdz[k][j][i][2]*dpos1_c[2])+Ext_E[2]
                                       };
                    // for(int c=0;c<3;c++) qdt2_2mE[c]=0 ;
                    px=qdt_m*(B[k][j][i][0]+ dBdx[k][j][i][0]*dpos1_c[0]+ dBdy[k][j][i][0]*dpos1_c[1]+dBdz[k][j][i][0]*dpos1_c[2]+Ext_B[0]);
                    py=qdt_m*(B[k][j][i][1]+ dBdx[k][j][i][1]*dpos1_c[0]+ dBdy[k][j][i][1]*dpos1_c[1]+dBdz[k][j][i][1]*dpos1_c[2]+Ext_B[1]);
                    pz=qdt_m*(B[k][j][i][2]+ dBdx[k][j][i][2]*dpos1_c[0]+ dBdy[k][j][i][2]*dpos1_c[1]+dBdz[k][j][i][2]*dpos1_c[2]+Ext_B[2]);
                    // px=0;
                    // py=0;
                    // pz=0;
                    pxx=px*px;
                    pxy=px*py;
                    pxz=px*pz;
                    pyy=py*py;
                    pyz=py*pz;
                    pzz=pz*pz;
                    r_determinant = 1.0/(1.0+pxx+pyy+pzz);
                    M11=r_determinant*(1.0 +pxx);
                    M12=r_determinant*(pz  +pxy);
                    M13=r_determinant*(pxz -py );
                    M21=r_determinant*(pxy -pz );
                    M22=r_determinant*(1.0 +pyy);
                    M23=r_determinant*(pyz +px );
                    M31=r_determinant*(pxz +py );
                    M32=r_determinant*(pyz -px );
                    M33=r_determinant*(1.0 +pzz);
                    dx=2*pos1[p][n][0]-pos0[p][n][0]-pz*pos0[p][n][1]+py*pos0[p][n][2];
                    dy=2*pos1[p][n][1]-pos0[p][n][1]-px*pos0[p][n][2]+pz*pos0[p][n][0];
                    dz=2*pos1[p][n][2]-pos0[p][n][2]-py*pos0[p][n][0]+px*pos0[p][n][1];
                    for (int c=0; c<3; c++) pos0[p][n][c]=pos1[p][n][c];
                    pos1[p][n][0]=M11*dx+M12*dy+M13*dz +qdt2_2mE[0];
                    pos1[p][n][1]=M21*dx+M22*dy+M23*dz +qdt2_2mE[1];
                    pos1[p][n][2]=M31*dx+M32*dy+M33*dz +qdt2_2mE[2];
                    //stop particle
                    if(1)
                    {
                        for (int c=0; c<3; c++)
                        {
                            if(pos1[p][n][c]>posH[c])
                            {
                                pos1[p][n][c]=posH[c];
                                pos0[p][n][c]=posH[c];
                                q[p][n]=0;
                                jj=ncalc[p];
                            }
                            if(pos1[p][n][c]<posL[c])
                            {
                                pos1[p][n][c]=posL[c];
                                pos0[p][n][c]=posL[c];
                                q[p][n]=0;
                                jj=ncalc[p];
                            }
                        }
                    }
                }
            }

        }
    }

// print out one last time
    std::cerr << i<<": time = " <<t <<" s" <<", ne = "<<nt[0]<<", ni = "<<nt[1];
//set fields=0 in preparation

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cerr << "Time difference = " <<(float)( std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count())/1000 << "[s]" << std::endl;
    return 0;
}
