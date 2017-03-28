#ifndef FTRL_MAPDB_H
#define FTRL_MAPDB_H

#include<vector>
#include<string>
#include<unordered_map>
#include "modelUnit.h"
#include "virtualDB.h"
#include "utils.h"

class MapDB: public VirtualDB{
public:
    ModelUnit* getOrInitDB(std::string k){
        std::unordered_map<std::string, ModelUnit*>::iterator iterator = vMap.find(k);
        if (iterator == vMap.end()) {
            mtx.lock();
            ModelUnit* modelUnit = new ModelUnit();
            vMap.insert(make_pair(k, modelUnit));
            mtx.unlock();
            return modelUnit;
        }
        else {
            return iterator->second;
        }
    }

    double get(std::string k){
        std::unordered_map<std::string, double>::iterator iterator = vWeight.find(k);
        if (iterator == vWeight.end())
            return 0;
        else
            return iterator->second;
    }

    void print(std::ofstream& out){
        for (std::unordered_map<std::string, ModelUnit*>::iterator it = vMap.begin(); it != vMap.end(); ++it) {
            out << it->first << "\t" 
            << it->second->w <<"\t"<<it->second->g <<"\t"
            <<it->second->s <<"\t"<< it->second->z <<"\t"<< it->second->n <<std::endl;
        }
    }
    bool loadModel(std::ifstream& in){
        std::string line;
        //double fp=0;
        std::vector<std::string> r;
        std::vector<double> dList;
        while(getline(in,line)){
            r.clear();
            dList.clear();
            utils::splitString(line, '\t', &r);
            if (r.size() != 6) { // key,W,G,S,Z,N
                std::cout<< "Wrong format :" <<line<<std::endl;
                //return false;
                return true;
            }
            std::string key=r[0];
            int index = 0;
            try {
                for(int i=1;i<6;i++){
                    index = i;
                    dList.push_back(std::stod(r[i]));
                }
                ModelUnit *modelUnit= new ModelUnit(dList[0], dList[1], dList[2], dList[3], dList[4]);
                vMap.insert(std::make_pair(key,modelUnit));
                std::cout << "LOAD:" << key << std::endl;
            }catch(const std::exception &e){
                std::cout << "exception @loadModel : " << e.what() << " index: "<< index << " value: " << r[index] 
                << " line: "<< line << std::endl;
            }
        }
        return true;
    }

    bool loadNonZeroWeight(std::ifstream& in){
        std::string line;
        //double fp=0;
        std::vector<std::string> r;
        int i = 0;
        while(getline(in,line)){
            i++;
            r.clear();
            utils::splitString(line, '\t', &r);
            if (r.size() != 6) { // key,W,G,S,Z,N
                std::cout<<"Line: "<<i << " content:" <<line<<std::endl;
                if(r.size() != 0) {
                    return false;
                }
            }else {
                if(fabs(std::stod(r[1])) > smallDouble || getBiasKey() == r[0])
                    vWeight.insert(std::make_pair(r[0], std::stod(r[1])));
            }
        }
        return true;
    }

    MapDB(){}
    
    std::string& getBiasKey(){
        return bias;
    }
    
    bool isBiasInModel(){
        auto iter1 = vMap.find(getBiasKey());
        if (iter1 != vMap.end()) {
            return true;
        }
        auto iter2 = vWeight.find(getBiasKey());
        if (iter2 != vWeight.end()) {
            return true;
        }
        return false;
    }
private:
    std::unordered_map<std::string, ModelUnit*> vMap;
    std::unordered_map<std::string, double> vWeight;
    std::mutex mtx;
    std::string bias = "bias";
};
#endif //FTRL_MAPDB_H
