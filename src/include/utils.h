#ifndef FTRL_UTILS_H
#define FTRL_UTILS_H

#include <string>
#include <vector>

#include <cmath>
#include <limits>

class utils {
public:
	static const uint64_t MAX = std::numeric_limits<uint64_t>::max();

    void static splitString(std::string& line,char delimiter, std::vector<std::string>* r);
    double static sigmoid(double a);
    int static sgn(double x);

    uint64_t static hash(const char* str){
    	uint64_t seed = 131; // 31 131 1313 13131 131313 etc..
    	uint64_t hash = 0;

    	while (*str){
        	hash = hash * seed + (*str++);
    	}
    	return ((MAX * hash) & 0x7FFFFFFFFFFFFFFF);
  	}
};


#endif //FTRL_UTILS_H
