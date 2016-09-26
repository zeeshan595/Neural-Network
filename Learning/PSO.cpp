#include "PSO.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <cmath>

Learning::ParticleSwarmOptimisation::ParticleSwarmOptimisation(double (*meanSquaredError)(std::vector<std::vector<double> > data, std::vector<double> weights), int weightsLen)
{
    this->meanSquaredError = meanSquaredError;
	this->weightsLen = weightsLen;

	min = -10.0;
	max = +10.0;

	inertia_weight 		= 0.729;
	cognitive_weight 	= 1.49445;
	social_weight 		= 1.49445;
}

Learning::ParticleSwarmOptimisation::~ParticleSwarmOptimisation()
{

}

std::vector<double> Learning::ParticleSwarmOptimisation::Train(std::vector<std::vector<double> > trainData, int particles, double exitError, double deathProbability, int repeat)
{
    //Setup
	std::srand(std::time(0));
	int r = 0; //Repeat
	double r1, r2;// Randomisation
	std::vector<Particle> swarm;
	swarm.resize(particles);
	std::vector<double> bestGlobalPosition(weightsLen);
	double bestGlobalError = std::numeric_limits<double>::max();

	//Create Particles
	for (int i = 0; i < particles; i++)
	{
		std::vector<double> randomPosition(weightsLen);
		for (int j = 0; j < weightsLen; j++)
		{
			randomPosition[j] = (max - min) * ((double)rand() / RAND_MAX) + min;
		}
		double error = meanSquaredError(trainData, randomPosition);

		std::vector<double> randomVelocity(weightsLen);
		for (int j = 0; j < weightsLen; j++)
		{
			double low  = 0.1 * min;
			double high = 0.1 * max;
            randomVelocity[j] = (high - low) * ((double)rand() / RAND_MAX) + low;
		}
		swarm[i] = Particle(randomPosition, randomVelocity, error);
	}
	
	//Randomise Sequence
	std::vector<int> sequence(particles);
	for (int i = 0; i < particles; i++)
	{
		sequence[i] = i;
	}

	//Start Trainning Loop
	while (r < repeat)
	{

		//Stop If Error Is To Low
		if (bestGlobalError < exitError)
			break;
		
		//sequence = Shuffle(sequence);
		std::vector<double> newVelocity(weightsLen);
		std::vector<double> newPosition(weightsLen);
		double newError;

		for (int pi = 0; pi < particles; pi++)
		{
			//Calculate Velocity
			int i = sequence[pi];
			for (int j = 0; j < weightsLen; j++)
			{
				r1 = (double)rand() / RAND_MAX;
				r2 = (double)rand() / RAND_MAX;
				newVelocity[j] = 	((inertia_weight  * swarm[i].velocity[j]) +
									(cognitive_weight 	* r1 * (swarm[i].bestPosition[j] 	- swarm[i].position[j])) +
									(social_weight 		* r2 * (bestGlobalPosition[j] 		- swarm[i].position[j])));
			}
			swarm[i].velocity = newVelocity;
		
			//Calculate Position
			for (int j = 0; j < weightsLen; j++)
			{
				newPosition[j] = swarm[i].position[j] + newVelocity[j];
				//Keep position in range
				if (newPosition[j] < min)
					newPosition[j] = min;
				else if (newPosition[j] > max)
					newPosition[j] = max;
			}
			swarm[i].position = newPosition;

			//Get Best Error/Position
			newError = meanSquaredError(trainData, newPosition);
			swarm[i].error = newError;
			if (newError < swarm[i].bestError)
			{
				swarm[i].bestPosition = newPosition;
				swarm[i].bestError = newError;
			}
			if (newError < bestGlobalError)
			{
				bestGlobalError = newError;
				bestGlobalPosition = newPosition;
			}

			//Use Particle Death Probability And Kill Particles Randomly
			double die = (double)rand() / RAND_MAX;
			if (die < deathProbability)
			{
				//Get the worst particle
				int target;
				double worstError = 0;
				for (int j = 0; j < particles; j++)
				{
					if (worstError < swarm[j].error)
					{
						worstError = swarm[j].error;
						target = i;
					}
				}

				//Get 2 random particles
				int rand1 = std::rand() % particles;
				int rand2 = std::rand() % particles;

				//Cross over particles position
				for (int j = 0; j < weightsLen; j++)
				{
					if ((int)(std::rand() % 2) == 0)
						newPosition[j] = swarm[rand1].position[j];
					else
						newPosition[j] = swarm[rand2].position[j];
				}

				//mutate the position random amount
				int mutationSize = std::rand() % weightsLen;
				for (int j = 0; j < weightsLen; j++)
				{
					if (mutationSize == 0)
						break;
					
					if (int(std::rand() % 2) == 0)
					{
						newPosition[j] += (max - min) * ((double)rand() / RAND_MAX) + min;
						mutationSize--;
					}
				}
				
				//Update particles position
				swarm[target].position = newPosition;
			}
		}

		{
			//Display Progress
			if (r != 0)
				std::cout << "\x1b[A";
			
			std::cout << "Progress: " << (int)(((double)r / (double)repeat) * 100.0) << "  Error: " << bestGlobalError << std::endl;
		}

		r++;
	}

	return bestGlobalPosition;
}

std::vector<int> Shuffle(std::vector<int> sequence)
{
	std::srand(std::time(0));
	for (unsigned int i = 0; i < sequence.size(); ++i)
	{
		int r = std::rand() % sequence.size() + i;
		int tmp = sequence[r];
		sequence[r] = sequence[i];
		sequence[i] = tmp;
	}
	return sequence;
}