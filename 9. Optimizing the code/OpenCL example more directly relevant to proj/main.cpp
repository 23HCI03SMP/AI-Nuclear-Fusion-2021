#include "traj.h"
unsigned int n_space_div[3]= {n_space_divx,n_space_divy,n_space_divz};

int main()
{
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();
    omp_set_num_threads(nthreads);
    cl::Context context;
    cl::Device default_device;
    cl::Program program;
    cl_start(context,default_device, program);

    double t=0;
    int AA[1] = {-1};
    #pragma omp target
    {
        AA[0] = omp_is_initial_device();
    }
    if (!AA[0])
    {
        cout<<"Able to use GPU offloading with OMP!\n";
    }
    else
    {
        cout<<"\nNo GPU on OMP\n";
    }
    //float pos2[2][n_partd][3];
    auto *pos0x=new float[2][n_partd];
    auto *pos0y=new float[2][n_partd];
    auto *pos0z=new float[2][n_partd];
    auto *pos1x=new float[2][n_partd];
    auto *pos1y=new float[2][n_partd];
    auto *pos1z=new float[2][n_partd];
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
//    auto *precalc_r=new float[n_space_divz][n_space_divy][n_space_divx];
//    auto *precalc_r2=new float[n_space_divz][n_space_divy][n_space_divx];
    auto *precalc_r3=new float[n_space_divz][n_space_divy][n_space_divx];
    auto *np= new int32_t[2][n_space_divz][n_space_divy][n_space_divx];
//    auto *It= new int[2][n_space_divz][3];
    auto *KE= new float[2][n_output_part];
    int nt[2]= {0,0};

    auto qdt2_2mEx=new float[n_partd];
    auto qdt2_2mEy=new float[n_partd];
    auto qdt2_2mEz=new float[n_partd];
    auto px=new float[n_partd];
    auto py=new float[n_partd];
    auto pz=new float[n_partd];

    ofstream o_file[2];
    ofstream E_file,B_file;

    cout << std::scientific;
    cout.precision(2);
    cerr << std::scientific;
    cerr.precision(2);
    cout<<"float size="<<sizeof(float)<<", " <<"int size="<<sizeof(int)<<endl;
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

    cerr << "Start up Time difference = " << (float)(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count())/1000 << "[s]" << endl;
    begin = chrono::steady_clock::now();
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
            //optimized: loop vectorized using 8 byte vectors
            //cube plasma
            /*           pos0x[p][n]=gsl_ran_flat(rng,posLp[0],posHp[0]);
                       pos1x[p][n]=pos0x[p][n]+(gsl_ran_gaussian(rng,sigma[p])+v0[p][0])*dt[p];
                       pos0y[p][n]=gsl_ran_flat(rng,posLp[1],posHp[1]);
                       pos1y[p][n]=pos0y[p][n]+(gsl_ran_gaussian(rng,sigma[p])+v0[p][1])*dt[p];
                       pos0z[p][n]=gsl_ran_flat(rng,posLp[2],posHp[2]);
                       pos1z[p][n]=pos0z[p][n]+(gsl_ran_gaussian(rng,sigma[p])+v0[p][2])*dt[p];
            */
//cylinder plasma
            float r=posHp[0]*sqrt(gsl_ran_flat(rng,0,1));
            double x,y;
            gsl_ran_dir_2d(rng, &x, &y);
            pos0x[p][n]=r*x;
            pos1x[p][n]=pos0x[p][n]+(gsl_ran_gaussian(rng,sigma[p])+v0[p][0])*dt[p];
            pos0y[p][n]=r*y;
            pos1y[p][n]=pos0y[p][n]+(gsl_ran_gaussian(rng,sigma[p])+v0[p][1])*dt[p];
            pos0z[p][n]=gsl_ran_flat(rng,posLp[2],posHp[2]);
            pos1z[p][n]=pos0z[p][n]+(gsl_ran_gaussian(rng,sigma[p])+v0[p][2])*dt[p];

            //          if (n==0) cout << "p = " <<p <<", sigma = " <<sigma[p]<<", temp = " << Temp[p] << ",mass of particle = " << mp[p] << dt[p]<<endl;
            q[p][n]=qs[p];
            m[p][n]=mp[p];
            nt[p]+=q[p][n];
        }
    }

    gsl_rng_free (rng);                       // dealloc the rng

    cout <<"v0 electron = "<<v0[0][0]<<","<<v0[0][1]<<","<<v0[0][2]<<endl;
//get limits and spacing of Field cells

    cerr << " Set initial random positions: Time difference = " << (float)(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count())/1000 << "[s]" << endl;
    begin = chrono::steady_clock::now();
    float posL[3],posH[3];

    for (int c=0; c<3; c++) dd[c]=8*a0/(n_space-1);
    for (int c=0; c<3; c++)
    {
        posL[c]=-dd[c]*(n_space_div[c]-1.0)/2.0;
        posH[c]= dd[c]*(n_space_div[c]-1.0)/2.0;
        cout<<posL[c]<<","<<posH[c]<<","<<dd[c]<<endl;
    }

//print initial conditions
    {
        std::cout << "electron Temp = " <<Temp[0]<< " K, electron Density = "<< Density_e<<" m^-3" << endl;
        std::cout << "Plasma Frequency(assume cold) = " <<plasma_freq<< " Hz, Plasma period = "<< plasma_period<<" s" << endl;
        std::cout << "Cyclotron period = "<<Tcyclotron<<" s, Time for electron to move across 1 cell = "<<Tv<<" s" << endl;
        std::cout << "electron thermal velocity = "<<vel_e<<endl;
        std::cout << "dt = "<<dt[0]<<" s,"<<endl;
        std::cout << "Debye Length = " <<Debye_Length<< " m, initial dimension = "<< a0<<" m" << endl;
        std::cout << "number of particle per cell = " <<n_partd/(n_space*n_space*n_space)*8<< endl;

        E_file.open ("info.csv");
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
    for (unsigned int ii=0; ii<n_space_divx; ii++)
    {
        float rx2= ii*dd[0];
        rx2*=rx2;
        for (unsigned int jj=0; jj<n_space_divy; jj++)
        {
            float ry2= jj*dd[1];
            ry2*=ry2;
            for (unsigned int kk=0; kk<n_space_divz; kk++)
            {
                //loop vectorized using 32 byte vectors
                float rz2= kk*dd[2];
                rz2*=rz2;
                if (((ii==0)&(jj==0)&(jj==0)))
                {
                    // assume cells only effect other cells
                    precalc_r3[kk][jj][ii]=0;
                }
                else
                {
                    precalc_r3[kk][jj][ii]=1/sqrt(rx2+ry2+rz2);
//                   precalc_r[kk][jj][ii]=precalc_r3[kk][jj][ii];
                    precalc_r3[kk][jj][ii]*=precalc_r3[kk][jj][ii];
                    precalc_r3[kk][jj][ii]*=precalc_r3[kk][jj][ii];
                }
            }
        }
    }
//    save_vti("precalc", 0,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],1,t,(reinterpret_cast<const char*>(precalc_r)),"Float32");
    int i_time;
    for ( i_time=0; i_time<ndatapoints; i_time++)
    {
        cerr << i_time<<": time = " <<t <<" s" <<", ne = "<<nt[0]<<", ni = "<<nt[1];
        cerr << " Time difference = " << (float)(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count())/1000 << "[s]" << std::endl;
        begin = chrono::steady_clock::now();
        Ext_E[0]=0;
        Ext_E[1]=0;
        Ext_E[2]=0;
        Ext_B[0]=0;
        Ext_B[1]=0;
        Ext_B[2]=0;
        //set fields=0 in preparation
//       #pragma omp  distribute parallel for simd
//       for (unsigned int i=0; i<2*n_space_div[0]*3; i++) (reinterpret_cast< float*>(It))[i]=0;
        //      #pragma omp  distribute parallel for simd
        for (unsigned int i=0; i< n_space_div[0]*n_space_div[1]*n_space_div[2]*3; i++)
        {
            (reinterpret_cast< float*>(B))[i]=0;
            (reinterpret_cast< float*>(E))[i]=0;
            (reinterpret_cast< float*>(currentj[0]))[i]=0;
            (reinterpret_cast< float*>(currentj[1]))[i]=0;
        }
//         cout<<"E and B"<<endl;
//        #pragma omp  distribute parallel for simd
        for (unsigned int i=0; i<n_space_div[0]*n_space_div[1]*n_space_div[2]*2; i++) (reinterpret_cast< int32_t*>(np))[i]=0;
        cout<<"np"<<endl;
//        #pragma omp  distribute parallel for simd
//        for (unsigned int i=0; i<n_space_div[0]*n_space_div[1]*n_space_div[2]*2*3; i++) (reinterpret_cast< float*>(currentj))[i]=0;
        //      cout<<"currentdensity"<<endl;
        nt[0]=0;
        nt[1]=0;
        // find number of particle and current density fields
        for (int p=0; p<2; p++)
        {
            //    #pragma omp  parallel for  reduction(+: np[p],nt[p],currentj[p])
//            #pragma omp  parallel for  reduction(+: np[p][k][j][i],nt[p],currentj[p][k][j][i])
            for(int n=0; n<n_partd; n++)
            {
                //optimized: loop vectorized using 32 byte vectors
                unsigned int i=(unsigned int)((pos1x[p][n]-posL[0])/dd[0]+.5);
                unsigned int j=(unsigned int)((pos1y[p][n]-posL[1])/dd[1]+.5);
                unsigned int k=(unsigned int)((pos1z[p][n]-posL[2])/dd[2]+.5);
                np[p][k][j][i]+=q[p][n]; //number of charge (in units of 1.6e-19 C  in each cell
                nt[p]+=q[p][n];
                //current density p=0 electron j=nev in each cell n in units 1.6e-19 C m/s
                currentj[p][k][j][i][0]+=q[p][n]*(pos1x[p][n]-pos0x[p][n])/dt[p];
                //It[p][k][0]+=q[p][n]*(pos1x[p][n]-pos0x[p][n])/dt[p];
                currentj[p][k][j][i][1]+=q[p][n]*(pos1y[p][n]-pos0y[p][n])/dt[p];
                //It[p][k][2]+=q[p][n]*(pos1y[p][n]-pos0y[p][n])/dt[p];
                currentj[p][k][j][i][2]+=q[p][n]*(pos1z[p][n]-pos0z[p][n])/dt[p];
                //It[p][k][2]+=q[p][n]*(pos1z[p][n]-pos0z[p][n])/dt[p];
            }
        }

        //find E field must work out every i,j,k depends on charge in every other cell
        for (unsigned int i=0; i<n_space_divx; i++)
        {
            for (unsigned int j=0; j<n_space_divy; j++)
            {
                for (unsigned int k=0; k<n_space_divz; k++)
                {
                    //               #pragma omp  parallel for reduction(+: E[k][j][i],B[k][j][i])
                    for (unsigned int ii=0; ii<n_space_divx; ii++)
                    {
                        float ddd[3];
                        ddd[0]=((float)ii-(float)i)*dd[0];
                        int iii=abs((int)ii-(int) i);
                        for (unsigned int jj=0; jj<n_space_divy; jj++)
                        {
                            ddd[1]=((float)jj-(float)j)*dd[1];
                            int jjj=abs((int)jj-(int)j);

                            for (unsigned int kk=0; kk<n_space_divz; kk++)
                            {
                                int kkk=abs((int)kk-(int)k);
                                ddd[2]=((float)kk-(float)k)*dd[2];
                                for (int c=0; c<3; c++)
                                {
                                    if (intEon)   E[k][j][i][c] -=(np[1][kk][jj][ii]+np[0][kk][jj][ii])*precalc_r3[kkk][jjj][iii]*ddd[c];
                                }
                                if (intBon)
                                {
                                    float jc[3]= {currentj[1][kk][jj][ii][0]+currentj[0][kk][jj][ii][0],currentj[1][kk][jj][ii][1]+currentj[0][kk][jj][ii][1], currentj[1][kk][jj][ii][2]+currentj[0][kk][jj][ii][2]};
                                    B[k][j][i][0] -=(jc[1]*ddd[2]-jc[2]*ddd[1])*precalc_r3[kkk][jjj][iii];
                                    B[k][j][i][1] +=(jc[0]*ddd[2]-jc[2]*ddd[0])*precalc_r3[kkk][jjj][iii];
                                    B[k][j][i][2] -=(jc[0]*ddd[1]-jc[1]*ddd[0])*precalc_r3[kkk][jjj][iii];
                                }
                            }
                        }
                    }
                    for (int c=0; c<3; c++)
                    {
                        E[k][j][i][c]*=Vconst;
                        B[k][j][i][c]*=Aconst;
                        if  (B[k][j][i][c]>Bmax ) cout <<"B"<<int(B[k][j][i][c]/Bmax)<<",";
                    }
                }
            }
        }

        for (unsigned int i=1; i<n_space_divx-1; i++)
        {
            for (unsigned int j=1; j<n_space_divy-1; j++)
            {
                #pragma omp  distribute parallel for simd
                for (unsigned int k=1; k<n_space_divz-1; k++)
                {
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
        cerr << " calc E&B Time difference = " << (float)(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count())/1000 << "[s]" << endl;
        begin = chrono::steady_clock::now();
// print out internal Electric potential
//        save_vti("V", i,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],1,t,(reinterpret_cast<const char*>(Vfield)),"Float32");
        save_vti("E", i_time,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(E)),"Float32",sizeof(float));
// print out internal electron number
        save_vti("Ne", i_time,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],1,t,(reinterpret_cast<const char*>(np[0])),"Int32",sizeof(int));
// print out internal magnetic potential file
//   save_vti("A", i_time,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(Afield)),"Float32"),8;
        save_vti("B", i_time,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(B)),"Float32",sizeof(float));
// print out internal electron current density file
        save_vti("je", i_time,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(currentj[0])),"Float32",sizeof(float));
//print out some particle positions
        {
            for (int p=0; p<2; p++)
            {
                #pragma omp distribute parallel for simd
                for(int nprt=0; nprt<n_output_part; nprt++)
                {
                    int n=nprt*nprtd;
                    float dpos,dpos2=0;

                    dpos=(pos1x[p][n]-pos0x[p][n]);
                    dpos*=dpos;
                    dpos2+=dpos;
                    dpos=(pos1y[p][n]-pos0y[p][n]);
                    dpos*=dpos;
                    dpos2+=dpos;
                    dpos=(pos1z[p][n]-pos0z[p][n]);
                    dpos*=dpos;
                    dpos2+=dpos;
                    KE[p][nprt]=0.5*m[p][n]*(dpos2)/(e_charge_mass*dt[p]*dt[p]);
                    //in units of eV
                    int c=0;
                    posp[p][nprt][c]=pos0x[p][n];
                    c++;
                    posp[p][nprt][c]=pos0y[p][n];
                    c++;
                    posp[p][nprt][c]=pos0z[p][n];
                }
            }
            save_vtp("e", i_time,  n_output_part,1, t, (reinterpret_cast<const char*>(&KE[0][0])), (reinterpret_cast<const char*>(&posp[0][0][0])));
            save_vtp("d", i_time,  n_output_part,1, t, (reinterpret_cast<const char*>(&KE[1][0])), (reinterpret_cast<const char*>(&posp[1][0][0])));
        }
        cerr << " print data: Time difference = " << (float)(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count())/1000 << "[s]" << endl;
        begin = chrono::steady_clock::now();
        t+=dt[0]*ncalc[0];
        //work out motion
        for (int p=0; p<2; p++)
        {
            float qdt_m[2];
            qdt_m[0]=(float)qs[p]*e_charge_mass*dt[p]/(float)mp[p];
            qdt_m[1]=qdt_m[0]*dt[p];
//            cout<<p<<endl;
            for (int jj=0; jj<ncalc[p]; jj++)
            {
                #pragma omp distribute parallel for simd
                for(int n=0; n<n_partd; n++)
                {
                    //         if (n%10000==0) cout<<qdt_m<<","<<qdt2_2m<<endl;
                    float dpos1_l[3]= {pos1x[p][n]-posL[0],pos1y[p][n]-posL[1],pos1z[p][n]-posL[2]};
                    unsigned int i=((pos1x[p][n]-posL[0])/dd[0]+.5);
                    unsigned int j=((pos1y[p][n]-posL[1])/dd[1]+.5);
                    unsigned int k=((pos1z[p][n]-posL[2])/dd[2]+.5);

                    float dpos1_c[3]= {dpos1_l[0]-i*dd[0],dpos1_l[1]-j*dd[1],dpos1_l[2]-k*dd[2]};
                    qdt2_2mEx[n]= (E[k][j][i][0]+ dEdx[k][j][i][0]*dpos1_c[0]+ dEdy[k][j][i][0]*dpos1_c[1]+dEdz[k][j][i][0]*dpos1_c[2])+Ext_E[0];
                    qdt2_2mEy[n]= (E[k][j][i][1]+ dEdx[k][j][i][1]*dpos1_c[1]+ dEdy[k][j][i][1]*dpos1_c[1]+dEdz[k][j][i][1]*dpos1_c[2])+Ext_E[1];
                    qdt2_2mEz[n]= (E[k][j][i][2]+ dEdx[k][j][i][2]*dpos1_c[2]+ dEdy[k][j][i][2]*dpos1_c[1]+dEdz[k][j][i][2]*dpos1_c[2])+Ext_E[2];
                    px[n]=(B[k][j][i][0]+ dBdx[k][j][i][0]*dpos1_c[0]+ dBdy[k][j][i][0]*dpos1_c[1]+dBdz[k][j][i][0]*dpos1_c[2]+Ext_B[0]);
                    py[n]=(B[k][j][i][1]+ dBdx[k][j][i][1]*dpos1_c[0]+ dBdy[k][j][i][1]*dpos1_c[1]+dBdz[k][j][i][1]*dpos1_c[2]+Ext_B[1]);
                    pz[n]=(B[k][j][i][2]+ dBdx[k][j][i][2]*dpos1_c[0]+ dBdy[k][j][i][2]*dpos1_c[1]+dBdz[k][j][i][2]*dpos1_c[2]+Ext_B[2]);
                }
            }
            calc_pnn(px,py,pz,qdt2_2mEx,qdt2_2mEy,qdt2_2mEz,pos0x[p],pos0y[p],pos0z[p],&pos1x[p][0],&pos1y[p][0],&pos1z[p][0],qdt_m,n_partd, context,default_device,program);
            //              cout<<"ocl"<<p<<endl;
            #pragma omp distribute parallel for simd
            for(int n=0; n<n_partd; n++)
            {
                if(pos1x[p][n]>posH[0])
                {
                    pos1x[p][n]=posH[0];
                    pos0x[p][n]=posH[0];
                    q[p][n]=0;
                }
                if(pos1x[p][n]<posL[0])
                {
                    pos1x[p][n]=posL[0];
                    pos0x[p][n]=posL[0];
                    q[p][n]=0;
                }
                if(pos1y[p][n]>posH[1])
                {
                    pos1y[p][n]=posH[1];
                    pos0y[p][n]=posH[1];
                    q[p][n]=0;
                }
                if(pos1y[p][n]<posL[1])
                {
                    pos1y[p][n]=posL[1];
                    pos0y[p][n]=posL[1];
                    q[p][n]=0;
                }
                if(pos1z[p][n]>posH[2])
                {
                    pos1z[p][n]=posH[2];
                    pos0z[p][n]=posH[2];
                    q[p][n]=0;
                }
                if(pos1z[p][n]<posL[2])
                {
                    pos1z[p][n]=posL[2];
                    pos0z[p][n]=posL[2];
                    q[p][n]=0;
                }
            }
        }
    }
// print out one last time

    //set fields=0 in preparation
//       #pragma omp  distribute parallel for simd
//       for (unsigned int i=0; i<2*n_space_div[0]*3; i++) (reinterpret_cast< float*>(It))[i]=0;
//   #pragma omp  distribute parallel for simd
    for (unsigned int i=0; i< n_space_div[0]*n_space_div[1]*n_space_div[2]*3; i++)
    {
        (reinterpret_cast< float*>(B))[i]=0;
        (reinterpret_cast< float*>(E))[i]=0;
        (reinterpret_cast< float*>(currentj[0]))[i]=0;
        (reinterpret_cast< float*>(currentj[1]))[i]=0;
    }
//         cout<<"E and B"<<endl;
//    #pragma omp  distribute parallel for simd
    for (unsigned int i=0; i<n_space_div[0]*n_space_div[1]*n_space_div[2]*2; i++) (reinterpret_cast< int32_t*>(np))[i]=0;
//       cout<<"np"<<endl;
//        #pragma omp  distribute parallel for simd
//        for (unsigned int i=0; i<n_space_div[0]*n_space_div[1]*n_space_div[2]*2*3; i++) (reinterpret_cast< float*>(currentj))[i]=0;
    //      cout<<"currentdensity"<<endl;
    nt[0]=0;
    nt[1]=0;
    // find number of particle and current density fields
    for (int p=0; p<2; p++)
    {
//        #pragma omp  parallel for  reduction(+: np[p],nt[p],currentj[p])
        for(int n=0; n<n_partd; n++)
        {
            //optimized: loop vectorized using 32 byte vectors
            unsigned int i=(unsigned int)((pos1x[p][n]-posL[0])/dd[0]+.5);
            unsigned int j=(unsigned int)((pos1y[p][n]-posL[1])/dd[1]+.5);
            unsigned int k=(unsigned int)((pos1z[p][n]-posL[2])/dd[2]+.5);
            np[p][k][j][i]+=q[p][n]; //number of charge (in units of 1.6e-19 C  in each cell
            nt[p]+=q[p][n];
            //current density p=0 electron j=nev in each cell n in units 1.6e-19 C m/s
            currentj[p][k][j][i][0]+=q[p][n]*(pos1x[p][n]-pos0x[p][n])/dt[p];
            //It[p][k][0]+=q[p][n]*(pos1x[p][n]-pos0x[p][n])/dt[p];
            currentj[p][k][j][i][1]+=q[p][n]*(pos1y[p][n]-pos0y[p][n])/dt[p];
            //It[p][k][2]+=q[p][n]*(pos1y[p][n]-pos0y[p][n])/dt[p];
            currentj[p][k][j][i][2]+=q[p][n]*(pos1z[p][n]-pos0z[p][n])/dt[p];
            //It[p][k][2]+=q[p][n]*(pos1z[p][n]-pos0z[p][n])/dt[p];
        }
    }
    cout<<"number of particles"<<endl;
// find current
    /*
            //        for(int c=0; c<3; c++)
                    {
                        for (int i=0; i<n_space_div[2]; i++)
                        {
                                      cout <<It[0][i][2]*e_charge*r_part_spart/dd[2]<<",";
                        }
                            cout <<endl;
                    }
    */
    //find E field must work out every i,j,k depends on charge in every other cell
    for (unsigned int i=0; i<n_space_divx; i++)
    {
        for (unsigned int j=0; j<n_space_divy; j++)
        {
            for (unsigned int k=0; k<n_space_divz; k++)
            {
//                #pragma omp  parallel for reduction(+: E[k][j][i],B[k][j][i])
                for (unsigned int ii=0; ii<n_space_divx; ii++)
                {
                    float ddd[3];
                    ddd[0]=((float)ii-(float)i)*dd[0];
                    int iii=abs((int)ii-(int) i);
                    for (unsigned int jj=0; jj<n_space_divy; jj++)
                    {
                        ddd[1]=((float)jj-(float)j)*dd[1];
                        int jjj=abs((int)jj-(int)j);

                        for (unsigned int kk=0; kk<n_space_divz; kk++)
                        {
                            int kkk=abs((int)kk-(int)k);
                            ddd[2]=((float)kk-(float)k)*dd[2];
                            for (int c=0; c<3; c++)
                            {
                                if (intEon)   E[k][j][i][c] -=(np[1][kk][jj][ii]+np[0][kk][jj][ii])*precalc_r3[kkk][jjj][iii]*ddd[c];
                            }
                            if (intBon)
                            {
                                float jc[3]= {currentj[1][kk][jj][ii][0]+currentj[0][kk][jj][ii][0],currentj[1][kk][jj][ii][1]+currentj[0][kk][jj][ii][1], currentj[1][kk][jj][ii][2]+currentj[0][kk][jj][ii][2]};
                                B[k][j][i][0] -=(jc[1]*ddd[2]-jc[2]*ddd[1])*precalc_r3[kkk][jjj][iii];
                                B[k][j][i][1] +=(jc[0]*ddd[2]-jc[2]*ddd[0])*precalc_r3[kkk][jjj][iii];
                                B[k][j][i][2] -=(jc[0]*ddd[1]-jc[1]*ddd[0])*precalc_r3[kkk][jjj][iii];
                            }
                        }
                    }
                }
                for (int c=0; c<3; c++)
                {
                    E[k][j][i][c]*=Vconst;
                    B[k][j][i][c]*=Aconst;
                    if  (B[k][j][i][c]>Bmax ) cout <<"B"<<int(B[k][j][i][c]/Bmax)<<",";
                }
            }
        }
    }

    for (unsigned int i=1; i<n_space_divx-1; i++)
    {
        for (unsigned int j=1; j<n_space_divy-1; j++)
        {
            //      #pragma omp  distribute parallel for simd
            for (unsigned int k=1; k<n_space_divz-1; k++)
            {
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
    cerr << " calc E&B Time difference = " << (float)(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count())/1000 << "[s]" << endl;
    begin = chrono::steady_clock::now();
// print out internal Electric potential
//        save_vti("V", i,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],1,t,(reinterpret_cast<const char*>(Vfield)),"Float32");
    save_vti("E", i_time,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(E)),"Float32",sizeof(float));
// print out internal electron number
    save_vti("Ne", i_time,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],1,t,(reinterpret_cast<const char*>(np[0])),"Int32",sizeof(int));
// print out internal magnetic potential file
//   save_vti("A", i_time,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(Afield)),"Float32"),8;
    save_vti("B", i_time,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(B)),"Float32",sizeof(float));
// print out internal electron current density file
    save_vti("je", i_time,n_space_div, posL, dd, n_space_div[0]*n_space_div[1]*n_space_div[2],3,t,(reinterpret_cast<const char*>(currentj[0])),"Float32",sizeof(float));
//print out some particle positions
    {
        for (int p=0; p<2; p++)
        {
            #pragma omp distribute parallel for simd
            for(int nprt=0; nprt<n_output_part; nprt++)
            {
                int n=nprt*nprtd;
                float dpos,dpos2=0;

                dpos=(pos1x[p][n]-pos0x[p][n]);
                dpos*=dpos;
                dpos2+=dpos;
                dpos=(pos1y[p][n]-pos0y[p][n]);
                dpos*=dpos;
                dpos2+=dpos;
                dpos=(pos1z[p][n]-pos0z[p][n]);
                dpos*=dpos;
                dpos2+=dpos;
                KE[p][nprt]=0.5*m[p][n]*(dpos2)/(e_charge_mass*dt[p]*dt[p]);
                //in units of eV
                int c=0;
                posp[p][nprt][c]=pos0x[p][n];
                c++;
                posp[p][nprt][c]=pos0y[p][n];
                c++;
                posp[p][nprt][c]=pos0z[p][n];
            }
        }
        save_vtp("e", i_time,  n_output_part,1, t, (reinterpret_cast<const char*>(&KE[0][0])), (reinterpret_cast<const char*>(&posp[0][0][0])));
        save_vtp("d", i_time,  n_output_part,1, t, (reinterpret_cast<const char*>(&KE[1][0])), (reinterpret_cast<const char*>(&posp[1][0][0])));
    }


    cerr << i_time<<": time = " <<t <<" s" <<", ne = "<<nt[0]<<", ni = "<<nt[1];
//set fields=0 in preparation
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cerr << " Time difference = " <<(float)( chrono::duration_cast<chrono::milliseconds>(end - begin).count())/1000 << "[s]" << endl;
    return 0;
}
