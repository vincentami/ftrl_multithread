#include <sstream>
#include "../include/ftrl.h"

using std::pair;
using namespace std;

#define TOKEN           ('\t')
#define TOKEN_INTER     (':')
#define MODELNAME       ("MODEL#")  

const int cap = 15000;
const int log_num =200000;

static  int trainNo = 0;

bool FTRL::init(int type){
    sem_init(&sem,0,0);
    sem_init(&semPro,0,1);
    this->type = type;
    threadVec.clear();
    threadVec.push_back(std::thread(&FTRL::inputThread,this));
    for(int i=0;i< threadNum;i++){
        if(type == 0){
            threadVec.push_back(std::thread(&FTRL::predictThread,this));
        }
        else{
            threadVec.push_back(std::thread(&FTRL::trainThread,this));
        }
    }
    return true;
}


void FTRL::run(int type){
    if(init(type)){
        for(unsigned int i=0;i< threadVec.size();i++){
            threadVec[i].join();
        }
    }
}

void FTRL::inputThread(){
    std::string line;
    int line_num = 0;
    int i=0;
    bool finished_flag = false;
    while(true){
        sem_wait(&semPro);
        queueMtx.lock();
        for(i=0;i<cap;i++){
            if(!getline(std::cin,line)){
                finished_flag = true;
                break;
            }
            line_num++;
            lineQueue.push(line);
            if(line_num%log_num==0)
            {
                if(type == 1)
                    std::cout<<line_num<<" line has finished" << trainNo << ":"<<(trainNo*100/(line_num+1)) <<std::endl;
            }
        }
        queueMtx.unlock();
        sem_post(&sem);
        if(finished_flag)
            break;
    }
}


void FTRL::trainThread(){
    bool finished_flag = false;
    bool parseSuc = false;
    EntityUnit* entity=new EntityUnit();
    std::vector<std::string> input_vec;
    input_vec.reserve(cap);
    while(true){
        input_vec.clear();
        sem_wait(&sem);
        queueMtx.lock();
        for(int i=0;i<cap;i++){
            if(lineQueue.empty()){
                finished_flag = true;
                break;
            }
            input_vec.push_back(lineQueue.front());
            lineQueue.pop();
        }
        queueMtx.unlock();
        sem_post(&semPro);
        int trainNum = 0;
        for(unsigned int i=0;i<input_vec.size();i++){
            parseSuc = parseLineToEntity(input_vec[i], entity);
            if (parseSuc) {
                trainNum++;
                train(entity->feature,entity->label);
            }
        }

        trainNo += trainNum;
        if(finished_flag)
            break;
    }
    sem_post(&sem);
    queueMtx.unlock();
    delete entity;
}

void FTRL::predictThread(){
    bool finished_flag = false;
    EntityUnit* entity=new EntityUnit();
    std::vector<std::string> input_vec;
    std::vector<std::string> output_vec;
    input_vec.reserve(cap);
    output_vec.reserve(cap);
    bool parseSuc = false;
    while(true){
        input_vec.clear();
        output_vec.clear();
        sem_wait(&sem);
        queueMtx.lock();
        for(int i=0;i<cap;i++){
            if(lineQueue.empty()){
                finished_flag = true;
                break;
            }
            input_vec.push_back(lineQueue.front());
            lineQueue.pop();
        }
        queueMtx.unlock();
        sem_post(&semPro);
        for(unsigned int i=0;i<input_vec.size();i++){
            parseSuc = parseLineToEntity(input_vec[i], entity);
            if(parseSuc) {
                double p = predict(entity->feature);
                output_vec.push_back(std::to_string(entity->label) + " " + std::to_string(p));
            }
        }

        outMtx.lock();
        for(unsigned int i=0;i<output_vec.size();i++){
            fPredict << output_vec[i] <<std::endl;
        }
        outMtx.unlock();
        if(finished_flag)
            break;
    }
    sem_post(&sem);
    queueMtx.unlock();
    delete entity;
}

/*void FTRL::train(const std::vector<pair<std::string, double> >& fea, int label) {
    std::vector<ModelUnit*> tempvec(fea.size(),NULL);
    double p = 0.0;
    for (int i = 0; i < fea.size(); ++i) {
        const std::string& index = fea[i].first;
        tempvec[i] = WGSZN->getOrInitDB(index);
        ModelUnit& modelUnit = *(tempvec[i]);
        modelUnit.mtx.lock();
        if(fabs(modelUnit.z.load()) <= lambda1) {
            modelUnit.w.store(0.0);
        } else {
            modelUnit.w.store((-1) *
                              (1 / (lambda2 + (beta + sqrt(modelUnit.n.load())) / alpha)) *
                              (modelUnit.z.load() - utils::sgn(modelUnit.z.load()) * lambda1));
        }
        p += modelUnit.w.load() * fea[i].second;
        modelUnit.mtx.unlock();
    }
    p = utils::sigmoid(p);
    for (int i = 0; i < fea.size(); ++i) {
        ModelUnit& modelUnit = *(tempvec[i]);
        modelUnit.mtx.lock();
        modelUnit.g.store((p-label) * (fea[i].second));
        modelUnit.s.store(1 / alpha * (sqrt(modelUnit.n.load() + modelUnit.g.load() * modelUnit.g.load()) - sqrt(modelUnit.n.load())));
        modelUnit.z.store(modelUnit.z.load() + modelUnit.g.load() - modelUnit.s.load() * modelUnit.w.load());
        modelUnit.n.store(modelUnit.n.load() + modelUnit.g.load() * modelUnit.g.load());
        modelUnit.mtx.unlock();
    }
}*/

void FTRL::train(const std::vector<pair<std::string, double> >& fea, int label) {
    std::vector<ModelUnit*> tempvec(fea.size(),NULL);
    double p = 0.0;
    for (unsigned int i = 0; i < fea.size(); ++i) {
        const std::string& index = fea[i].first;
        tempvec[i] = WGSZN->getOrInitDB(index);

        ModelUnit& modelUnit = *(tempvec[i]);
        modelUnit.mtx.lock();
        if(fabs(modelUnit.z) <= lambda1) {
            modelUnit.w = 0.0;
        } else {
            modelUnit.w = (-1) *
                              (1 / (lambda2 + (beta + sqrt(modelUnit.n)) / alpha)) *
                              (modelUnit.z - utils::sgn(modelUnit.z) * lambda1);
        }
        p += modelUnit.w * fea[i].second;
        modelUnit.mtx.unlock();
    }

    p = utils::sigmoid(p);
    for (unsigned int i = 0; i < fea.size(); ++i) {
        ModelUnit& modelUnit = *(tempvec[i]);
        modelUnit.mtx.lock();
        modelUnit.g = (p-label) * (fea[i].second);
        modelUnit.s = 1 / alpha * (sqrt(modelUnit.n + modelUnit.g * modelUnit.g) - sqrt(modelUnit.n));
        modelUnit.z += modelUnit.g - modelUnit.s * modelUnit.w;
        modelUnit.n += modelUnit.g * modelUnit.g;
        modelUnit.mtx.unlock();
    }
}


double FTRL::predict(const std::vector<pair<std::string, double> >& fea) {
    double res = 0.0;
    for (unsigned int i = 0; i != fea.size(); ++i) {
        /*std::cout << "predict::"<< fea[i].first << 
        ":"<< WGSZN->get(fea[i].first) <<std::endl;*/
        res += fea[i].second * (WGSZN->get(fea[i].first));
        //res += fea[i].second * (WGSZN->getOrInitDB(fea[i].first)->w);
    }
    return utils::sigmoid(res);
}

 bool splitLabel(const std::string &line, std::vector<string> &splitRes, 
                    int &label, std::string &fename){
    stringstream ss;
    string item = "";
    char delim = '\t';

    std::vector<string> vec;
    splitRes.reserve(1024);
    ss.str(line);
    int i = 0;
    while (std::getline(ss, item, delim)){
        i++;
        if (!(item.length() == 1 && isblank(item.at(0))) 
            && (item.length() != 0)){
            vec.push_back(item);
            cout << i << ":" << item <<endl;
        }
        item.clear();
    } 

    if (vec.size() != 14){
        /*if(splitRes.size() == 14){
            cout << "splitRes.size:" << splitRes[14] << endl;
        }else{
            cout << "splitRes.size()" << splitRes.size() << endl;
        }*/
        return false;
    }

    label = atoi(vec[0].c_str());
    if (!(label == 0 ||label == 1)){
        cout << "labe " << label << endl;
        cout << "wrong "<< vec[0]<< ":"<<vec[1]<< endl;
        return false;  
    }

    ss.str(vec[13]); 
    item = "";
    delim = ' ';
    i = 0;
    while (std::getline(ss, item, delim)){
        i++;
        if (!(item.length() == 1 && isblank(item.at(0))) 
            && (item.length() != 0)){
            splitRes.push_back(item);
            cout << i << ":" << item <<endl;
        }
        item.clear();
    } 

    fename = "10^1^121";

    std::cout << "tokensize:" << splitRes.size() << std::endl;
    /*
    string modelName(splitRes[0]);
    fename = modelName;

    splitRes.erase(splitRes.begin());*/

    return true;
}

bool FTRL::parseLineToEntity(const std::string& line, EntityUnit *entity) {
    entity->feature.clear();
    std::string key;
    double value;
    int label;
    uint32_t i;
    bool  ret;
    uint64_t hashVal ; 
    string featureLabel ;
    std::size_t posb;
    std::vector<string> featureLabelVec;
    std::string fename;
    std::stringstream ss;
    try {
        ret = splitLabel(line, featureLabelVec, label, fename);
        if(ret == false){
            return false;
        }

        entity->label = label > 0 ? 1: 0;

        for (i = 0; i < featureLabelVec.size() ; i++){
            featureLabel = featureLabelVec.at(i);
            posb = featureLabel.find_first_of(TOKEN_INTER, 0);
            if (posb == std::string::npos){
                cout << "wrong feature format! " << i << " : "<<featureLabel.length() << ":" << featureLabel << endl;
                continue;
            }

            key = featureLabel.substr(0, posb);
            if (key.compare(WGSZN->getBiasKey()) == 0){
                std::cout << "input should not contains bias key: " << key << "\n" << line << std::endl;
                return false;
            }

            value = stod(featureLabel.substr(posb+1,featureLabel.length()));
            if(value!=0) {
                //hashVal = utils::hash((key + "#" + fename).c_str());
                //ss << hashVal << "#" << fename;
                entity->feature.push_back(std::make_pair(key,value));
                //ss.str(std::string());
            }
            //cout << "key: value-> " << key  << ":" << value << endl; 
        }

        if(addBias){
            key = WGSZN->getBiasKey();
            value = 1.0;
            //hashVal = utils::hash(("WO#"+ fename).c_str());
            //ss << hashVal << "#" << fename;
            entity->feature.push_back(std::make_pair(key,value));
            //ss.str(std::string());
        }
    }catch (const std::exception &e){
        std::cout << "exception @parseLineToEntity : " << e.what() << " line: "<< line << std::endl;
        return false;
    }

    return true;
}

void FTRL::printW(std::ofstream& out) {
    WGSZN->print(out);
}

bool FTRL::loadModel(std::ifstream& fModel){
    if(!WGSZN->loadModel(fModel)){
        return false;
    }
    return (WGSZN->isBiasInModel() == addBias);
}
bool FTRL::loadNonZeroWeight(std::ifstream& fModel){
    if(!WGSZN->loadNonZeroWeight(fModel)){
        return false;
    }
    addBias = WGSZN->isBiasInModel();
    return true;
}
