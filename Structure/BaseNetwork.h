#ifndef _BASE_NETWORK
#define _BASE_NETWORK

#include "../Core/ActivationType.h"

#include <vector>

namespace Structure
{
    class BaseNetwork
    {
    public:
        virtual int GetInputs();
        virtual int GetOutputs();
        virtual std::vector<std::vector<double> > GetNodeValues();
        virtual std::vector<double> GetWeights();
        virtual int GetWeightsLength();
        virtual void SetWeights(std::vector<double> result);
        virtual std::vector<Core::ActivationType> GetActivations();
        virtual std::vector<std::vector<std::vector<double> > > GetOrderedWeights();
        virtual std::vector<std::vector<double> > GetOrderedBiases();
        
        virtual double MeanSquaredError(std::vector<std::vector<double> > data, std::vector<double> weights);
        virtual double MeanSquaredError(std::vector<std::vector<double> > data);
        virtual std::vector<double> Compute(std::vector<double> xValue);
    };
};

#endif