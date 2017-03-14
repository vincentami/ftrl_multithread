#ifndef FTRL_MODELUNIT_H
#define FTRL_MODELUNIT_H
#include <string>
#include <mutex>
#include <atomic>
/*struct ModelUnit{
    std::atomic<double> w;
    std::atomic<double> g;
    std::atomic<double> s;
    std::atomic<double> z;
    std::atomic<double> n;
    std::mutex mtx;
    ModelUnit(double w_i,double g_i,double s_i,double z_i,double n_i){w.store(w_i);g.store(g_i);s.store(s_i);z.store(z_i);n.store(n_i);};
    ModelUnit(){w.store(0.0);g.store(0.0);s.store(0.0);z.store(0.0);n.store(0.0);}
};*/

struct ModelUnit{
    double w;
    double g;
    double s;
    double z;
    double n;
    std::mutex mtx;
    ModelUnit(double w_i,double g_i,double s_i,double z_i,double n_i){w=w_i;g=g_i;s=s_i;z=z_i;n=n_i;};
    ModelUnit(){w=0.0;g=0.0;s=0.0;z=0.0;n=0.0;}
    //ModelUnit(){ModelUnit(0.0,0.0,0.0,0.0,0.0);}
};
#endif //FTRL_MODELUNIT_H
