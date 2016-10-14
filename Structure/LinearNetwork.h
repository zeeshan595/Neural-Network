#ifndef _LINEAR_NETWORK
#define _LINEAR_NETWORK

#include "BaseNetwork.h"
#include "../Core/Activation.h"
#include "../Core/ActivationType.h"

#include <vector>

namespace Structure
{
    class LinearNetwork: public BaseNetwork
    {
    public:
        int GetInputs();
        int GetOutputs();
        std::vector<int> GetLayers();
        std::vector<std::vector<double> > GetNodeValues();
        std::vector<double> GetWeights();
        int GetWeightsLength();
        void SetWeights(std::vector<double> result);
        std::vector<Core::ActivationType> GetActivations();
        std::vector<std::vector<std::vector<double> > > GetOrderedWeights();
        std::vector<std::vector<double> > GetOrderedBiases();

        void GenerateWeights(double MIN, double MAX);
        std::vector<double> Compute(std::vector<double> xValues);
        double MeanSquaredError(std::vector<std::vector<double> > data, std::vector<double> weights);
        double MeanSquaredError(std::vector<std::vector<double> > data);

        LinearNetwork(std::vector<int> layers, std::vector<Core::ActivationType> activations);

    private:
        std::vector<int> layers;
        std::vector<Core::ActivationType> activations;

        std::vector<std::vector<std::vector<double> > > weights;
        std::vector<std::vector<double> > biases;
        std::vector<std::vector<double> > node_values;
    };
};

#endif