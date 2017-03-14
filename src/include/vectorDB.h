
#ifndef FTRL_VECTORDB_H
#define FTRL_VECTORDB_H
#include <vector>
#include <string>
#include <cstddef>
#include <mutex>
#include "virtualDB.h"
#include "modelUnit.h"
#include "utils.h"

class VectorDB: public VirtualDB{
public:
    ModelUnit* getOrInitDB(std::string k){
        unsigned long ikey = std::stoul(k);
        if(vMap.size() <= ikey || vMap[ikey]==NULL){
            mtx.lock();
            while(vMap.size() <= ikey){
                vMap.push_back(NULL);
            }
            if(vMap[ikey]==NULL)
                vMap[ikey] = new ModelUnit();
            mtx.unlock();
        }
        return vMap[ikey];
    }

    double get(std::string k){
        unsigned long ikey = std::stoul(k);
        if(ikey < vWeight.size())
            return vWeight[ikey];
        else
            return 0;
    }

    void print(std::ofstream& out){
        for (unsigned int i=0;i< vMap.size();i++) {
            if(vMap[i] !=NULL)
                out << i << "\t" << vMap[i]->w <<"\t"<< vMap[i]->g <<"\t"<< vMap[i]->s <<"\t"<< vMap[i]->z <<"\t"<< vMap[i]->n <<std::endl;
        }
    }
    bool loadModel(std::ifstream& in){
        std::string line;
        //double fp=0;
        std::vector<std::string> r;
        std::vector<double> dlist;
        while(getline(in,line)){
            r.clear();
            dlist.clear();
            utils::splitString(line, '\t', &r);
            if (r.size() != 6) { // key,W,G,S,Z,N
                std::cout<<line<<std::endl;
                return false;
            }
            std::string key=r[0];
            for(int i=1;i<6;i++){
                dlist.push_back(std::stod(r[i]));
            }
            ModelUnit *modelUnit= new ModelUnit(dlist[0],dlist[1],dlist[2],dlist[3],dlist[4]);
            unsigned long ikey = std::stoul(key);
            while(vMap.size() <= ikey){
                vMap.push_back(NULL);
            }
            vMap[ikey] = modelUnit;
        }
        return true;
    }

    bool loadNonZeroWeight(std::ifstream& in){
        std::string line;
        //double fp=0;
        std::vector<std::string> r;
        while(getline(in,line)){
            r.clear();
            utils::splitString(line, '\t', &r);
            if (r.size() < 2) { // key,W,G,S,Z,N
                std::cout<<line<<std::endl;
                return false;
            }
            unsigned int ikey = std::stoul(r[0]);
            while(vWeight.size() <= ikey){
                vWeight.push_back(0);
            }
            if(fabs(std::stod(r[1])) > smallDouble || getBiasKey() == r[0])
                vWeight[ikey] = std::stod(r[1]);
            else
                vWeight[ikey] = 0;
        }
        return true;
    }


    VectorDB(int model_size){
        vMap.assign(model_size, NULL);
    }
    VectorDB(){}
    
    std::string& getBiasKey(){
        return bias;
    }
    
    bool isBiasInModel(){
        unsigned int ikey = std::stoul(getBiasKey());
        if (ikey < vMap.size() && NULL != vMap[ikey]) {
            return true;
        }
        if (ikey < vWeight.size() && 0 != vWeight[ikey]) {
            return true;
        }
        return false;
    }    
private:
    std::vector<ModelUnit*> vMap;
    std::vector<double> vWeight;
    std::mutex mtx;
    std::string bias="0";
};
#endif //FTRL_VECTORDB_H
