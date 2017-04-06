#ifndef _Multilayer_FeedForward_Network
#define _Multilayer_FeedForward_Network

class MFNN
{
public:
    struct Layer
    {
        std::vector<Neuron*>    neurons;
        std::vector<Synapse*>   synapsis;
    };

    MFNN(
        std::vector<uint32_t> neurons_per_layer
    );
    ~MFNN();

    std::vector<double> Compute(
        std::vector<double>     inputs
    );

    double ComputeMeanSquaredError(
        std::vector<std::vector<double> >   data,
        std::vector<double>                 weights
    );

    void SetWeights(
        std::vector<double>     weights
    );
    std::vector<double> GetWeights();

    Layer* GetLayer(
        uint32_t                layer_number
    );
    std::vector<double> GetOutput();

private:
    std::vector<Layer*>         layers;

    void SetRandomWeights();
};

#include "MFNN.cpp"
#endif