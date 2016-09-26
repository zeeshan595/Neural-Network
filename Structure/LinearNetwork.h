#ifndef _LINEAR_NETWORK
#define _LINEAR_NETWORK

#include "../Core/Activation.h"
#include "../Core/ActivationType.h"

#include <vector>

namespace Structure
{
    class LinearNetwork
    {
    public:
        std::vector<int> GetLayers();
        std::vector<Core::ActivationType> GetActivations();
        std::vector<double> GetWeights();
        void SetWeights(std::vector<double> w);
        int GetWeightsLength();


        LinearNetwork(std::vector<int> layers, std::vector<Core::ActivationType> activations);
        ~LinearNetwork();

        std::vector<double> Compute(std::vector<double> xValues);
        void GenerateWeights();

    private:
        std::vector<int> layers;
        std::vector<Core::ActivationType> activations;
    
        std::vector<std::vector<double> > biases;
        std::vector<std::vector<std::vector<double> > > weights;
    };
};

#endif