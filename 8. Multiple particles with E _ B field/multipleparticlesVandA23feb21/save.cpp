#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <omp.h>
#include <string>

#include "traj.h"

void save_vti(string filename, int i, \
              int n_space_div[3], double posl[3], double dd[3], uint64_t num,int ncomponents, double t, \
              const char* data,string typeofdata)
        {
            std::ofstream os("./out/"+filename+"_"+to_string(i)+".vti", std::ios::binary | std::ios::out);
            os<< "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\"> \n ";
            os<<"<ImageData WholeExtent=\"0 ";
            os<<to_string(n_space_div[0]-1)+" 0 " +to_string(n_space_div[1]-1)+" 0 "+to_string(n_space_div[2]-1)+"\" ";
            os<<"Origin=\""+to_string(posl[0])+" "+to_string(posl[1])+" "+to_string(posl[2])+"\"";
            os<<" Spacing=\""+to_string(dd[0])+" "+to_string(dd[1])+" "+to_string(dd[2])+"\" ";
            os<<"Direction=\"1 0 0 0 1 0 0 0 1\"> \n";
            os<< "<FieldData>\n";
            os<<  "<DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"appended\" RangeMin=\"0\" RangeMax=\"0\" offset=\"0\"/>\n";
            os<<"</FieldData>\n";
            os<<"<Piece Extent=\"0 ";
            os<<to_string(n_space_div[0]-1)+" 0 " +to_string(n_space_div[1]-1)+" 0 "+to_string(n_space_div[2]-1)+"\">\n";
            os<<"<PointData Scalars=\""+filename+"\">\n";
            os<<"<DataArray type=\""+typeofdata+"\" Name=\""+filename+"\" NumberOfComponents=\""+to_string(ncomponents)+"\" format=\"appended\" RangeMin=\"0\" RangeMax=\"0\" offset=\"16\" />\n";
            os<<"  </PointData>\n";
            os<<"<CellData>\n";
            os<<"  </CellData>\n";
            os<<"</Piece>\n";
            os<<"</ImageData>\n";
            os<<"<AppendedData encoding=\"raw\">_";
            uint64_t num1=8;
            os.write(reinterpret_cast<const char*>(&num1), std::streamsize(sizeof(num1)));
            os.write(reinterpret_cast<const char*>(&t), std::streamsize(sizeof(double)));
            num1=num*ncomponents*8;
            os.write(reinterpret_cast<const char*>(&num1), std::streamsize(sizeof(num1)));
            os.write(data, std::streamsize(num*ncomponents*sizeof(double)));
            os<<"</AppendedData>\n";
            os<<"</VTKFile>";
            os.close();
        }

void save_pvd(string filename, int ndatapoints)
    {
        std::ofstream os("./out/"+filename+".pvd", std::ios::binary | std::ios::out);
        os<<"<VTKFile type=\"Collection\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
        os<< "<Collection>\n";
        for (int i=0; i<ndatapoints; i++)
        {
            os <<"<DataSet timestep=\""+to_string(i)+"\" part=\"0\" file=\""+filename+"_"+to_string(i)+".vti\"/>\n";
        }
        os<< " </Collection>\n";
        os<<"</VTKFile>\n";
    }
