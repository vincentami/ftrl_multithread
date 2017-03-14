
#ifndef FTRL_ENTITYUNIT_H
#define FTRL_ENTITYUNIT_H
#include <string>
#include <vector>
struct EntityUnit{
public:
    std::string print(){
        std::string r = std::to_string(label);
        for(unsigned int i=0;i<feature.size();i++)
            r += " " + feature[i].first +":"+ std::to_string(feature[i].second);
        return r;
    }
    int label;
    std::vector<std::pair<std::string, double> > feature;
};
#endif //FTRL_ENTITYUNIT_H
