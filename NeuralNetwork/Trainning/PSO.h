#ifndef _PARTICLE_SWARM_OPTIMIZATION
#define _PARTICLE_SWARM_OPTIMIZATION

namespace PSO
{
    struct Particle
    {
        //Particle position
        std::vector<double>         position;
        //Particle velocity
        std::vector<double>         velocity;
        //Particle best position found
        std::vector<double>         best_position;
        //Particle error
        double                      error;
        //particle minimum error found
        double                      best_error;
    };

    void Train(
        //training data 
        std::vector<std::vector<double> >   train_data,
        //amount of particles to use for training
        uint32_t                            particles_count,
        //max epochs this training should be repeated for
        uint32_t                            repeat,
        //A pointer to the neural network that
        //needs to be trained.
        BaseNetwork*                        base_network
    );
};

#include "PSO.cpp"
#endif