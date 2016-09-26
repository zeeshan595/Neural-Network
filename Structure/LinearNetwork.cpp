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

    //Setup weights & biases
    //Make sure the input layer is not included so -1
    weights.resize(layers.size() - 1);
    biases.resize(layers.size() - 1);

    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        biases[i].resize(layers[i + 1]);

        weights[i].resize(layers[i + 1]);
        for (int j = 0; j < layers[i + 1]; j++)
        {
            weights[i][j].resize(layers[i]);
        }
    }

    GenerateWeights();
}

void Structure::LinearNetwork::GenerateWeights()
{
    std::srand(std::time(0));
    double low  = -0.01;
    double high = +0.01;

    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        for (int j = 0; j < layers[i + 1]; j++)
        {
            biases[i][j] = (high - low) * ((double)rand() / RAND_MAX) + low;

            for (int k = 0; k < layers[i]; k++)
            {
                weights[i][j][k] = (high - low) * ((double)rand() / RAND_MAX) + low;
            }
        }
    }
}

Structure::LinearNetwork::~LinearNetwork()
{

}

std::vector<double> Structure::LinearNetwork::Compute(std::vector<double> xValues)
{
    assert(xValues.size() == (unsigned int)layers[0]);

    std::vector<double> prev_result = xValues;
    std::vector<double> result;

    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        //Reset results
        result.resize(layers[i + 1]);
        for (int j = 0; j < layers[i + 1]; j++)
            result[j] = 0;

        //Compute Weights
        for (int j = 0; j < layers[i + 1]; j++)
        {
            for (int k = 0; k < layers[i]; k++)
            {
                result[j] += prev_result[k] * weights[i][j][k];
            }
        }

        //Add Biases
        for (int j = 0; j < layers[i + 1]; j++)
        {
            result[j] += biases[i][j];
        }

        //Apply Activation
        result = Core::Activation::ApplyActivation(result, activations[i + 1]);
        prev_result = result;
    }

    return result;
}

//-------------------------
//----Getters & Setters----
//-------------------------

std::vector<int> Structure::LinearNetwork::GetLayers()
{
    return layers;
}

std::vector<Core::ActivationType> Structure::LinearNetwork::GetActivations()
{
    return activations;
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

void Structure::LinearNetwork::SetWeights(std::vector<double> w)
{
    assert(w.size() == (unsigned int)GetWeightsLength());

    int L = 0;

    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        for (int j = 0; j < layers[i + 1]; j++)
        {
            biases[i][j] = w[L];
            L++;
        }
    }

    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        for (int j = 0; j < layers[i + 1]; j++)
        {
            for (int k = 0; k < layers[i]; k++)
            {                
                weights[i][j][k] = w[L];
                L++;
            }
        }
    }
}

int Structure::LinearNetwork::GetWeightsLength()
{
    int L = 0;

    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        for (int j = 0; j < layers[i + 1]; j++)
        {
            L++;
        }
    }

    for (unsigned int i = 0; i < layers.size() - 1; i++)
    {
        for (int j = 0; j < layers[i + 1]; j++)
        {
            for (int k = 0; k < layers[i]; k++)
            {                
                L++;
            }
        }
    }

    return L;
}