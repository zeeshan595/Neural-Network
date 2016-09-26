#ifndef _PSO
#define _PSO

#include <vector>

namespace Learning
{
    class ParticleSwarmOptimisation
    {
    public:
        double min;
        double max;

        double inertia_weight;
        double cognitive_weight;
        double social_weight;

        double (*meanSquaredError)(std::vector<std::vector<double> > data, std::vector<double> weights);
        ParticleSwarmOptimisation(double (*meanSquaredError)(std::vector<std::vector<double> > data, std::vector<double> weights), int weightsLen);
        ~ParticleSwarmOptimisation();

        std::vector<double> Train(std::vector<std::vector<double> > trainData, int particles, double exitError, double deathProbability, int repeat);

    private:
        int weightsLen;
        std::vector<int> Shuffle(std::vector<int> sequence);

        class Particle
        {
        public:
            std::vector<double> position;
            std::vector<double> velocity;

            std::vector<double> bestPosition;
            double error;
            double bestError;

            Particle(){}
            Particle(std::vector<double> position, std::vector<double> velocity, double error)
			{
				this->position = position;
				this->velocity = velocity;
				this->bestPosition = position;
				this->error = error;
				this->bestError = error;
			}

            ~Particle()
			{
				position.erase(position.begin(), position.end());
				velocity.erase(position.begin(), position.end());
				bestPosition.erase(position.begin(), position.end());
			}
        };
    };
};

#endif