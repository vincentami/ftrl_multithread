
#ifndef FTRL_FTRL_H
#define FTRL_FTRL_H
#include <set>
#include <cmath>
#include <string>
#include <vector>
#include <fstream>
#include <utility>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <semaphore.h>
#include "utils.h"
#include "modelUnit.h"
#include "vectorDB.h"
#include "entityUnit.h"
#include "mapDB.h"

class FTRL{
public:
    //  用作train时的初始化函数
    FTRL(double a, double b, double c, double d, int t_num, int model_size, bool b_addBias) {
        //    :alpha(a),beta(b),lambda1(c),lambda2(d), threadNum(t_num), addBias(b_addBias){
        alpha = a;
        beta = b;
        lambda1 = c;
        lambda2 = d;
        threadNum = t_num;
        model_size = model_size;
        addBias = b_addBias;            
        
        //WGSZN = new VectorDB(model_size);
        WGSZN = new MapDB();
    };
    // 用作test时的初始化函数
    FTRL(int t_num, std::string f_out){
        //WGSZN = new VectorDB();
        threadNum = t_num;
        WGSZN = new MapDB();
        fPredict.open(f_out,std::ofstream::out);
    };
    bool init(int type);
    void run(int type);
    void printW(std::ofstream& out);
    bool loadModel(std::ifstream& fModel);
    bool loadNonZeroWeight(std::ifstream& fModel);
    void printArgv(){
        std::cout <<"l1:" <<lambda1 << " l2:" << lambda2 << " al:" << alpha <<" be:"<<beta 
            <<" bias:" << addBias << std::endl;
    }

private:
    double alpha;
    double beta;
    double lambda1;
    double lambda2;
    int threadNum;
    bool addBias;
    int type;
    int model_size;

    std::ofstream fPredict;
    std::mutex queueMtx, outMtx;
    sem_t sem, semPro;
    std::queue<std::string> lineQueue;
    std::vector<std::thread> threadVec;
    VirtualDB* WGSZN;
    bool parseLineToEntity(const std::string& line, EntityUnit *unit);
    double predict(const std::vector<std::pair<std::string, double> >& fea);
    void train(const std::vector<std::pair<std::string, double> >& fea, int label);
    void trainThread();
    void predictThread();
    void inputThread();
};


#endif //FTRL_FTRL_H
