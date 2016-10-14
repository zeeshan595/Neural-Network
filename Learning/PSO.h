#ifndef _PSO
#define _PSO

#include "../Structure/BaseNetwork.h"

namespace Learning
{
    class ParticleSwarmOptimisation
    {
    public:
        bool auto_assign_best_value;
        double MIN;
        double MAX;

        double inertia_weight;
        double cognitive_weight;
        double social_weight;

        ParticleSwarmOptimisation(Structure::BaseNetwork* network);
        ~ParticleSwarmOptimisation();

        std::vector<double> Train(std::vector<std::vector<double> > train_data, int particles, double exit_error, double death_probability, int repeat);

    private:
        Structure::BaseNetwork* network;

        std::vector<int> Shuffle(std::vector<int> sequence);

        class Particle
        {
        public:
            std::vector<double> position;
            std::vector<double> velocity;

            std::vector<double> best_position;
            double error;
            double best_error;

            Particle(){}
            Particle(std::vector<double> position, std::vector<double> velocity, double error)
			{
				this->position = position;
				this->velocity = velocity;
				this->best_position = position;
				this->error = error;
				this->best_error = error;
			}

            ~Particle()
			{
				position.erase(position.begin(), position.end());
				velocity.erase(position.begin(), position.end());
				best_position.erase(position.begin(), position.end());
			}
        };
    };
};

#endif