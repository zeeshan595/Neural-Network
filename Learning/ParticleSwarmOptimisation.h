#ifndef _PSO
#define _PSO

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <limits>

namespace Learning
{
	class ParticleSwarmOptimisation
	{
	public:
		double minX;
		double maxX;

		double inertiaWeight;
		double cognitiveWeight;
		double socialWeight;		
	
		double (*meanSquaredError)(std::vector<std::vector<double> > data, std::vector<double> weights);
		ParticleSwarmOptimisation(double (*meanSquaredError)(std::vector<std::vector<double> > data, std::vector<double> weights), std::vector<double> weights);
		~ParticleSwarmOptimisation();

		std::vector<double> Train(std::vector<std::vector<double> > trainData, int particles, double exitError, double deathProbability, int repeat);
	private:
		std::vector<int> Shuffle(std::vector<int> sequence);
		std::vector<double> weights;

		class Particle
		{
		public:
			std::vector<double> position;
			std::vector<double> velocity;

			std::vector<double> bestPosition;
			double error;
			double bestError;
			
			Particle()
			{
				
			}

			Particle(std::vector<double> position, std::vector<double> velocity, double error)
			{
				this->position = position;
				this->velocity = velocity;
				this->bestPosition = position;
				this->error = error;
				this->bestError = error;
			}
		};
	};
};

#endif