#include <iostream>
#include <map>
#include <fstream>
#include "../include/ftrl.h"


struct Option {
    //Option() : alpha(0.05f), beta(1.0f), lambda1(0.1f), lambda2(5.0f),
    //           nr_threads(1), b_init(false), model_size(100000), b_addBias(false) {}
    std::string model_path;
    std::string init_m_path;
    double alpha = 0.05f;
    double beta = 1.0f;
    double lambda1 = 0.1f;
    double lambda2 = 5.0f;
    int nr_threads = 1;
    int model_size = 1000000;
    bool b_init = false;
    bool b_addBias = false;
};

std::string train_help() {
    return std::string(
            "\nusage: cat sample | ftrl_train_std [<options>]"
                    "\n"
                    "\n"
                    "options:\n"
                    "-m <model_path>: set the model path\n"
                    "-a <alpha>: the larger the larger steps of weight\tdefault:0.05\n"
                    "-b <beta>: the larger the smaller steps of weight\tdefault:1.0\n"
                    "-l1 <lambda1>: the larger the sparser\tdefault:0.1\n"
                    "-l2 <lambda2>: the larger the smaller steps of weight\tdefault:5.0\n"
                    "-core <nr_threads>: set the number of threads\tdefault:1\n"
                    "-im <initial_m>: set the initial value of model\n"
                    "-size <model_size>: set the largest size of model\n"
                    "-bias <1/0>: set the bias, the value can only be 1 or 0\tdefault:0\n"
    );
}

std::vector<std::string> argv_to_args(int const argc, char const * const * const argv) {
    std::vector<std::string> args;
    for(int i = 1; i < argc; ++i)
        args.push_back(std::string(argv[i]));
    return args;
}

Option parse_option(std::vector<std::string> const &args) {
    uint32_t const argc = static_cast<uint32_t>(args.size());
    std::cout<< train_help() <<"\n";
    Option opt;

    uint32_t i = 0;
    for(; i < argc; i++) {
        if(args[i].compare("-m") == 0) {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.model_path = args[++i];
        }

        else if(args[i].compare("-a") == 0) {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.alpha = std::stod(args[++i]);
        }
        else if(args[i].compare("-b") == 0) {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.beta = std::stod(args[++i]);
        }
        else if(args[i].compare("-l1") == 0) {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.lambda1 = std::stod(args[++i]);
        }

        else if(args[i].compare("-l2") == 0) {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.lambda2 = std::stod(args[++i]);
        }
        else if(args[i].compare("-core") == 0) {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.nr_threads = std::stoi(args[++i]);
        }
        else if(args[i].compare("-size") == 0) {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.model_size = std::stoi(args[++i]);
        }
        else if(args[i].compare("-im") == 0) {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            opt.init_m_path = args[++i];
            opt.b_init=true; //if im field exits , that means b_init = true !
        }
        else if(args[i].compare("-bias") == 0) {
            if(i == argc-1)
                throw std::invalid_argument("invalid command\n");
            std::string value = args[++i];
            if(value.compare("1") == 0) {
                opt.b_addBias = true;
            }
            else if(value.compare("0") == 0) {
                opt.b_addBias = false;
            }
            else {
                throw std::invalid_argument("invalid command\n");
            }
        }
        else {
            break;
        }
    }
    return opt;
}


int main(int argc,char* argv[]) {
    std::cin.sync_with_stdio(false);
    std::cout.sync_with_stdio(false);
    Option opt;
    try {
        opt = parse_option(argv_to_args(argc, argv));
    }
    catch(std::invalid_argument const &e) {
        std::cout << e.what();
        return EXIT_FAILURE;
    }

    FTRL modelObj(opt.alpha, opt.beta, opt.lambda1, opt.lambda2, opt.nr_threads, opt.model_size, opt.b_addBias);
    modelObj.printArgv();

    if(opt.b_init) {
        std::ifstream f_temp(opt.init_m_path.c_str());
        if(!modelObj.loadModel(f_temp)) {
            std::cout<<"wrong model"<<std::endl;
            return 0;
        }
        f_temp.close();
    }

    modelObj.run(1); // for train
    std::ofstream f_weight(opt.model_path.c_str(), std::ofstream::out);
    modelObj.printW(f_weight);
}


