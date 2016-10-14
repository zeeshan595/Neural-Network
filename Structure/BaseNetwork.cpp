#include "BaseNetwork.h"

int Structure::BaseNetwork::GetInputs()
{
    return -1;
}

int Structure::BaseNetwork::GetOutputs()
{
    return -1;
}

std::vector<std::vector<double> > Structure::BaseNetwork::GetNodeValues()
{
    std::vector<std::vector<double> > r;
    return r;
}

std::vector<double> Structure::BaseNetwork::GetWeights()
{
    std::vector<double> r;
    return r;
}

int Structure::BaseNetwork::GetWeightsLength()
{
    return -1;
}

void Structure::BaseNetwork::SetWeights(std::vector<double> result)
{

}

std::vector<double> Structure::BaseNetwork::Compute(std::vector<double> xValue)
{
    std::vector<double> r;
    return r;
}

double Structure::BaseNetwork::MeanSquaredError(std::vector<std::vector<double> > data)
{
    return -1;
}

double Structure::BaseNetwork::MeanSquaredError(std::vector<std::vector<double> > data, std::vector<double> weights)
{
    return -1;
}

std::vector<Core::ActivationType> Structure::BaseNetwork::GetActivations()
{
    std::vector<Core::ActivationType> r;
    return r;
}

std::vector<std::vector<std::vector<double> > > Structure::BaseNetwork::GetOrderedWeights()
{
    std::vector<std::vector<std::vector<double> > > r;
    return r;
}

std::vector<std::vector<double> > Structure::BaseNetwork::GetOrderedBiases()
{
    std::vector<std::vector<double> > r;
    return r;
}