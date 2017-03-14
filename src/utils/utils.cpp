
#include "utils.h"
#include <cmath>

const double kPrecision = 0.0000000001;

void utils::splitString(std::string& line, char delimiter, std::vector<std::string>* r){
    unsigned int begin=0;
    for(unsigned int i=0;i<line.size();i++){
        if(line[i] == delimiter){
            (*r).push_back(line.substr(begin, i-begin));
            begin=i+1;
        }
    }
    if(begin<line.size())
        (*r).push_back(line.substr(begin,line.size()-begin));
}

double utils::sigmoid(double a) {
    return 1 / (1 + exp(-1*a));
}

int utils::sgn(double x) {
    if (x > kPrecision)
        return 1;
    else
        return -1;
}