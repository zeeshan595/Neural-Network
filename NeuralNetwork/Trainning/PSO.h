#ifndef _PARTICLE_SWARM_OPTIMIZATION
#define _PARTICLE_SWARM_OPTIMIZATION

namespace PSO
{
    struct Particle
    {
        std::vector<double>         position;
        std::vector<double>         velocity;
        std::vector<double>         best_position;
        double                      error;
        double                      best_error;
    };

    void Train(
        std::vector<std::vector<double> >   train_data,
        uint32_t                            particles_count,
        double                              exit_error,
        double                              death_probability,
        uint32_t                            repeat,
        BaseNetwork*                        base_network
    );
};

#include "PSO.cpp"
#endif