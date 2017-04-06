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

    std::vector<double> Train(
        std::vector<std::vector<double> >   terain_data,
        uint32_t                            particles_count,
        double                              exit_error,
        double                              death_probability,
        uint32_t                            repeat,
        std::vector<double>                 current_weights,
        std::vector<double>                 (*compute_function) (std::vector<double>),
        double                              (*error_function)   (std::vector<std::vector<double> >, std::vector<double>)
    );

    std::vector<uint32_t> Shuffle(
        std::vector<uint32_t> sequence
    );
};

#include "PSO.cpp"
#endif