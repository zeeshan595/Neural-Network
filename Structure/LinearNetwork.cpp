#include "LinearNetwork.h"

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <ctime>

Structure::LinearNetwork::LinearNetwork(std::vector<int> layers, std::vector<Core::ActivationType> activations)
{
    assert(layers.size() > 1);
    assert(activations.size() == layers.size());
    assert(activations[0] == Core::ActivationType::NONE);

    this->layers = layers;
    this->activations = activations;

    //Setup
    biases.resize(layers.size() - 1);
    weights.resize(layers.size() - 1);
    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        biases[i].resize(layers[i + 1]);
        weights[i].resize(layers[i + 1]);

        for (int j = 0; j < layers[i + 1]; j++)
        {
            weights[i][j].resize(layers[i]);
        }
    }

    node_values.resize(layers.size());
    for (unsigned int i = 0; i < layers.size(); i++)
        node_values[i].resize(layers[i]);

    GenerateWeights(-0.01, 0.01);
}

std::vector<double> Structure::LinearNetwork::Compute(std::vector<double> xValues)
{
    assert(xValues.size() == (unsigned int)layers[0]);

    node_values[0] = xValues;

    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        //Setup
        for (int j = 0; j < layers[i + 1]; j++)
            node_values[i + 1][j] = 0;

        //Multiply Weights
        for (int j = 0; j < layers[i + 1]; j++)
        {
            for (int k = 0; k < layers[i]; k++)
            {
                node_values[i + 1][j] += (node_values[i][k] * weights[i][j][k]);
            }
        }

        //Add Biases
        for (int j = 0; j < layers[i + 1]; j++)
        {
            node_values[i + 1][j] += biases[i][j];
        }

        //Apply Activation
        node_values[i + 1] = Core::Activation::ApplyActivation(node_values[i + 1], activations[i + 1]);
    }

    return node_values[node_values.size() - 1];
}

void Structure::LinearNetwork::GenerateWeights(double MIN, double MAX)
{
    std::srand(std::time(0));

    for (unsigned int i = 0; i < weights.size(); i++)
    {
        for (unsigned int j = 0; j < weights[i].size(); j++)
        {
            biases[i][j] = (MAX - MIN) * ((double)std::rand() / RAND_MAX) + MIN;

            for (unsigned int k = 0; k < weights[i][j].size(); k++)
            {
                weights[i][j][k] = (MAX - MIN) * ((double)std::rand() / RAND_MAX) + MIN;
            }
        }
    }
}

double Structure::LinearNetwork::MeanSquaredError(std::vector<std::vector<double> > data, std::vector<double> weights)
{
    this->SetWeights(weights);
    return this->MeanSquaredError(data);
}

double Structure::LinearNetwork::MeanSquaredError(std::vector<std::vector<double> > data)
{
	std::vector<double> xValues(this->GetInputs()); // Inputs
	std::vector<double> tValues(this->GetOutputs()); //Outputs

	double sum_squared_error = 0.0;
	for (unsigned int i = 0; i < data.size(); ++i)
	{
		// assumes data has x-values followed by y-values
		std::copy(data[i].begin(), data[i].begin() + this->GetInputs(), xValues.begin());
		std::copy(data[i].begin() + this->GetInputs(), data[i].begin() + this->GetInputs() + this->GetOutputs(), tValues.begin());

		std::vector<double> yValues = this->Compute(xValues);
		for (unsigned int j = 0; j < yValues.size(); ++j)
			sum_squared_error += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
	}

	return sum_squared_error;
}

//-------------
//---Getters---
//-------------

int Structure::LinearNetwork::GetInputs()
{
    return layers[0];
}

int Structure::LinearNetwork::GetOutputs()
{
    return layers[layers.size() - 1];
}

std::vector<int> Structure::LinearNetwork::GetLayers()
{
    return layers;
}

std::vector<std::vector<double> > Structure::LinearNetwork::GetNodeValues()
{
    return node_values;
}

int Structure::LinearNetwork::GetWeightsLength()
{
    int k = 0;

    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        k += (layers[i + 1] * layers[i]) + layers[i + 1];
    }
    return k;
}

std::vector<double> Structure::LinearNetwork::GetWeights()
{
    std::vector<double> result(GetWeightsLength());
    int L = 0;

    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        for (int j = 0; j < layers[i + 1]; j++)
        {
            result[L] = biases[i][j];
            L++;
        }
    }

    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        for (int j = 0; j < layers[i + 1]; j++)
        {
            for (int k = 0; k < layers[i]; k++)
            {                
                result[L] = weights[i][j][k];
                L++;
            }
        }
    }

    return result;
}

void Structure::LinearNetwork::SetWeights(std::vector<double> result)
{
    assert(result.size() == (unsigned int)GetWeightsLength());

    int L = 0;

    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        for (int j = 0; j < layers[i + 1]; j++)
        {
            biases[i][j] = result[L];
            L++;
        }
    }

    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        for (int j = 0; j < layers[i + 1]; j++)
        {
            for (int k = 0; k < layers[i]; k++)
            {                
                weights[i][j][k] = result[L];
                L++;
            }
        }
    }
}

std::vector<Core::ActivationType> Structure::LinearNetwork::GetActivations()
{
    return activations;
}

std::vector<std::vector<std::vector<double> > > Structure::LinearNetwork::GetOrderedWeights()
{
    return weights;
}

std::vector<std::vector<double> > Structure::LinearNetwork::GetOrderedBiases()
{
    return biases;
}