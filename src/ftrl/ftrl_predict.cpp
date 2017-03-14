#include <iostream>
#include <fstream>
#include "../include/ftrl.h"
using std::pair;

int main(int argc,char* argv[]) {
    std::cin.sync_with_stdio(false);
    std::cout.sync_with_stdio(false);

    if(argc!=4){
        std::cout<<"wrong argc;"<<std::endl;
        std::cout<<"cat test | ./ftrl_predict modelFile threadNum outPath"<<std::endl;
        return 0;
    }

    std::ifstream fModel(argv[1]);
    int threadNum = std::stoi(argv[2]);
    FTRL modelObj(threadNum, argv[3]);

    if(!modelObj.loadNonZeroWeight(fModel)){
        std::cout<<"check your model"<<std::endl;
        return 0;
    }
    modelObj.run(0);
}

