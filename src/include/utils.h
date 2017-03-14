#ifndef FTRL_UTILS_H
#define FTRL_UTILS_H


#include <string>
#include <vector>
class utils {
public:
    void static splitString(std::string& line,char delimiter, std::vector<std::string>* r);
    double static sigmoid(double a);
    int static sgn(double x);
};


#endif //FTRL_UTILS_H
