#include "PSO.h"

#include <cassert>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <cmath>

Learning::ParticleSwarmOptimisation::ParticleSwarmOptimisation(Structure::BaseNetwork* network)
{
    this->network = network;
	auto_assign_best_value = true;

    MIN = -10.0;
    MAX = +10.0;

    inertia_weight      = 0.729;
    cognitive_weight    = 1.49445;
    social_weight       = 1.49445;
}

Learning::ParticleSwarmOptimisation::~ParticleSwarmOptimisation()
{

}

std::vector<double> Learning::ParticleSwarmOptimisation::Train(std::vector<std::vector<double> > train_data, int particles, double exit_error, double death_probability, int repeat)
{
	assert((int)train_data[0].size() == network->GetInputs() + network->GetOutputs());
	assert(particles > 0);
	assert(death_probability < 1 && death_probability > 0);
	assert(repeat > 0);
	assert(exit_error > 0);
	assert(train_data.size() > 0);

	//setup
	int weights_length = network->GetWeightsLength();
	std::vector<double> current_weights = network->GetWeights();

	int r = 0; //repeat
	std::srand(std::time(0));
	double r1, r2; //Randomisers
	std::vector<Particle> swarm(particles);
	std::vector<double> best_global_position(weights_length);
	double best_global_error = std::numeric_limits<double>::max();

	//Set 1st particle's position to current weights
	{
		double error = network->MeanSquaredError(train_data, current_weights);
		std::vector<double> random_velocity(weights_length);
		for (int j = 0; j < weights_length; j++)
		{
			double low  = 0.1 * MIN;
			double high = 0.1 * MAX;
			random_velocity[j] = (high - low) * ((double)std::rand() / RAND_MAX) + low;
		}
		swarm[0] = Particle(current_weights, random_velocity, error);

		if (error < best_global_error)
		{
			best_global_position = current_weights;
			best_global_error = error;
		}
	}

	//Create Particles and assign random postions & velocity
	for (int i = 1; i < particles; i++)
	{
		//Compute Random Position
		std::vector<double> random_position(weights_length);
		for (int j = 0; j < weights_length; j++)
		{
			random_position[j] = (MAX - MIN) * ((double)std::rand() / RAND_MAX) + MIN;
		}
		//Get Error
		double error = network->MeanSquaredError(train_data, random_position);

		//Compute Random Velocity
		std::vector<double> random_velocity(weights_length);
		for (int j = 0; j < weights_length; j++)
		{
			double low  = 0.1 * MIN;
			double high = 0.1 * MAX;
			random_velocity[j] = (high - low) * ((double)std::rand() / RAND_MAX) + low;
		}
		swarm[i] = Particle(random_position, random_velocity, error);

		if (error < best_global_error)
		{
			best_global_position = random_position;
			best_global_error = error;
		}
	}

	//Create a sequence to randomse particles in future
	std::vector<int> sequence(particles);
	for (int i = 0; i < particles; i++)
		sequence[i] = i;
	
	//Start Trainning Loop
	while (r < repeat)
	{
		//If error is too low then break loop
		if (best_global_error < exit_error)
			break;

		//Shuffle particles order using sequence
		sequence = Shuffle(sequence);

		//Setup
		std::vector<double> new_position(weights_length);
		std::vector<double> new_velocity(weights_length);
		double new_error;

		//go throw all particles
		for (int pi = 0; pi < particles; pi++)
		{
			//Select random particle
			int i = sequence[pi];

			//Calculate Velocity
			for (int j = 0; j < weights_length; j++)
			{
				r1 = (double)std::rand() / RAND_MAX;
				r2 = (double)std::rand() / RAND_MAX;
				new_velocity[j] = 	((inertia_weight  	* swarm[i].velocity[j]) +
									(cognitive_weight 	* r1 * (swarm[i].best_position[j] 	- swarm[i].position[j])) +
									(social_weight 		* r2 * (best_global_position[j] 	- swarm[i].position[j])));
			}
			swarm[i].velocity = new_velocity;

			//Calculate Position
			for (int j = 0; j < weights_length; j++)
			{
				new_position[j] = swarm[i].position[j] + new_velocity[j];
				//keep position in range
				if (new_position[j] < MIN)
					new_position[j] = MIN;
				else if (new_position[j] > MAX)
					new_position[j] = MAX;
			}
			swarm[i].position = new_position;

			//Get Error
			new_error = network->MeanSquaredError(train_data, new_position);
			swarm[i].error = new_error;

			//Compare with best error
			if (new_error < swarm[i].best_error)
			{
				swarm[i].best_position = new_position;
				swarm[i].best_error = new_error;
			}
			//Compare with best global error
			if (new_error < best_global_error)
			{
				
				best_global_position = new_position;
				best_global_error = new_error;
			}
			//Randomly kill particles
			if (((double)std::rand() / RAND_MAX) < death_probability)
			{
				//Reset particle and intilise it again
				for (int j = 0; j < weights_length; j++)
				{
					swarm[i].position[j] = (MAX - MIN) * ((double)std::rand() / RAND_MAX) - MIN;
				}
				swarm[i].error = network->MeanSquaredError(train_data, swarm[i].position);
				swarm[i].best_position = swarm[i].position;
				swarm[i].best_error = swarm[i].error;

				if (swarm[i].error < best_global_error)
				{
					best_global_error = swarm[i].error;
					best_global_position = swarm[i].position;
				}
			}
		}

		//Display Learning Progress
		if (r > 0)
			std::cout << "\x1b[A";
		std::cout << "Progress: " << (int)(((double)r / repeat) * 100.0) << " | Error: " << best_global_error << std::endl;

		r++;
	}
    
	if (auto_assign_best_value)
		network->SetWeights(best_global_position);
	else
		network->SetWeights(current_weights);

	return best_global_position;
}

std::vector<int> Learning::ParticleSwarmOptimisation::Shuffle(std::vector<int> sequence)
{
    std::vector<int> result = sequence;
	std::srand(std::time(0));
	for (unsigned int i = 0; i < result.size(); ++i)
	{
		int r = std::rand() % result.size();
		int tmp = result[r];
		result[r] = result[i];
		result[i] = tmp;
	}
	return result;
}