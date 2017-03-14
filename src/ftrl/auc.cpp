#include "fstream"
#include "algorithm"
#include "iostream"
#include "../include/utils.h"


void readFile(std::vector<int>& labels, std::vector<double>& probs){
    std::string line;
    std::vector<std::string> sp;
    while(std::getline(std::cin, line)){
        sp.clear();
        utils::splitString(line, ' ', &sp);
        if(std::stoi(sp[0]) <= 0)
            labels.push_back(0);
        else
            labels.push_back(1);
        probs.push_back(std::stof(sp[1]));
    }
}

double scoreAuc(std::vector<int>& labels, std::vector<double>& probs){

    long double aucTemp = 0.0;
    long double tp = 0.0;
    long double tpPre = 0.0;
    long double fp = 0.0;
    long double fpPre = 0.0;
    //unsigned long long p = 0;
    //unsigned long long n = 0;

    std::vector<int> range;
    range.reserve(probs.size());
    for(unsigned int i=0;i< probs.size();i++){
        range.push_back(i);
    }
    std::sort(range.begin(),range.end(),[&probs](int i1, int i2){return probs[i1] > probs[i2];});

    double last_prob = probs[range[0]] + 1.0;

    for(unsigned int i =0; i < probs.size(); i++){
        if(last_prob != probs[range[i]]){
            aucTemp += (tp + tpPre) * (fp - fpPre) / 2.0;
            tpPre = tp;
            fpPre = fp;
            last_prob = probs[range[i]];
        }
        if(labels[range[i]] == 1)
            tp++;
        else
            fp ++;
    }

    aucTemp += (tp + tpPre) * (fp - fpPre) / 2.0;
    double auc = aucTemp / tp / fp;
    return auc;
}

int main(int argc,char* argv[]) {
    std::vector<int> labels;
    std::vector<double> probs;
    readFile(labels, probs);
    double auc = scoreAuc(labels, probs);
    std::cout << "auc:" << auc << std::endl;
}
