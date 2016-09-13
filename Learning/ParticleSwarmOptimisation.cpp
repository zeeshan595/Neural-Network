#include "ParticleSwarmOptimisation.h"

using namespace Learning;

ParticleSwarmOptimisation::ParticleSwarmOptimisation(double (*meanSquaredError)(std::vector<std::vector<double> > data, std::vector<double> weights), std::vector<double> weights)
{
	minX = -10.0;
	maxX =  10.0;
	inertiaWeight   = 0.729;
	cognitiveWeight = 1.49445;
	socialWeight    = 1.49445;
	this->meanSquaredError = meanSquaredError;
	this->weights = weights;
}
ParticleSwarmOptimisation::~ParticleSwarmOptimisation()
{
	weights.erase(weights.begin(), weights.begin() + weights.size());
}

std::vector<double> ParticleSwarmOptimisation::Train(std::vector<std::vector<double> > trainData, int particles, double exitError, double deathProbability, int repeat)
{
	if (weights.size() < 1)
	{
		std::cout << "Weights length cannot be less then 1" << std::endl;
		throw 20;
	}
	//Setup
	std::srand(std::time(0));
	int weightsLen = weights.size();
	int r = 0; //Repeat
	double r1, r2;// Randomisation
	std::vector<Particle> swarm;
	swarm.resize(particles);
	std::vector<double> bestGlobalPosition;
	bestGlobalPosition.resize(weightsLen);
	double bestGlobalError = std::numeric_limits<double>::max();

	//Create Particles
	for (int i = 0; i < particles; i++)
	{
		std::vector<double> randomPosition;
		randomPosition.resize(weightsLen);
		for (int j = 0; j < weightsLen; j++)
		{
			double randomNumber = (double)rand() / RAND_MAX;
			randomPosition[j] = (maxX - minX) * randomNumber + minX;
		}
		double error = meanSquaredError(trainData, randomPosition);

		std::vector<double> randomVelocity;
		randomVelocity.resize(weightsLen);
		for (int j = 0; j < weightsLen; j++)
		{
			double low  = 0.1 * minX;
			double high = 0.1 * maxX;
			double randomNumber = (double)rand() / RAND_MAX;
			randomVelocity[j] = (high - low) * randomNumber + low;
		}
		swarm[i] = Particle(randomPosition, randomVelocity, error);
	}

	//Randomise Sequence
	std::vector<int> sequence;
	sequence.resize(particles);
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
		std::vector<double> newVelocity;
		newVelocity.resize(weightsLen);
		std::vector<double> newPosition;
		newPosition.resize(weightsLen);
		double newError;

		for (int pi = 0; pi < particles; pi++)
		{
			//Calculate Velocity
			int i = sequence[pi];
			for (int j = 0; j < weightsLen; j++)
			{
				r1 = (double)rand() / RAND_MAX;
				r2 = (double)rand() / RAND_MAX;
				newVelocity[j] = (inertiaWeight * swarm[i].velocity[j]) +
					(cognitiveWeight * r1 * (swarm[i].bestPosition[j] - swarm[i].position[j])) +
					(socialWeight * r2 * (bestGlobalPosition[j] - swarm[i].position[j]));
			}
			swarm[i].velocity = newVelocity;
		
			//Calculate Position
			for (int j = 0; j < weightsLen; j++)
			{
				newPosition[j] = swarm[i].position[j] + newVelocity[j];
				//Keep position in range
				if (newPosition[j] < minX)
					newPosition[j] = minX;
				else if (newPosition[j] > maxX)
					newPosition[j] = maxX;
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
				//New Position, Leave Velocity, Update error
				for (int j = 0; j < weightsLen; j++)
				{
					swarm[i].position[j] = (maxX - minX) * ((double)rand() / RAND_MAX) + minX;
				}
				swarm[i].error = meanSquaredError(trainData, swarm[i].position);
				swarm[i].bestPosition = swarm[i].position;
				swarm[i].bestError = swarm[i].error;

				if (swarm[i].error < bestGlobalError)
				{
					bestGlobalError = swarm[i].error;
					bestGlobalPosition = swarm[i].position;
				}
			}
		}

		if (r != 0)
			std::cout << "\x1b[A";
		
		std::cout << "Progress: " << ((double)(r + 1) / (double)repeat) * 100.0 << std::endl;

		r++;
	}

	return bestGlobalPosition;
}

std::vector<int> ParticleSwarmOptimisation::Shuffle(std::vector<int> sequence)
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