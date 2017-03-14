//
// Created by liyongbao on 15-7-22.
//

#ifndef FTRL_VIRTUALDB_H
#define FTRL_VIRTUALDB_H
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include "modelUnit.h"

class VirtualDB{
public:
    virtual void print(std::ofstream& out) = 0;
    virtual bool loadModel(std::ifstream& in) = 0;
    virtual bool loadNonZeroWeight(std::ifstream& in) = 0;
    virtual ModelUnit* getOrInitDB(std::string k) = 0;
    virtual double get(std::string k) = 0;
    virtual std::string& getBiasKey() = 0;
    virtual bool isBiasInModel() = 0;
    double smallDouble = 0.0000001;
};
#endif //FTRL_VIRTUALDB_H
