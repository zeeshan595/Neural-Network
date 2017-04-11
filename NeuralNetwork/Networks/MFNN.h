#ifndef _Multilayer_FeedForward_Network
#define _Multilayer_FeedForward_Network

class MFNN : public BaseNetwork
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

    double GetMeanSquaredError(
        std::vector<std::vector<double> >   data,
        std::vector<double>                 weights
    );
    double GetAccuracy(
        std::vector<std::vector<double> > data
    );

    void SetWeights(
        std::vector<double>     weights
    );
    std::vector<double> GetWeights();

    Layer* GetLayer(
        uint32_t                layer_number
    );
    std::vector<double> GetOutput();

    void SetRandomWeights();

    void TrainUsingBP(
        std::vector<std::vector<double> >       train_data,
        double                                  learning_rate,
        double                                  momentum,
        double                                  weight_decay,
        uint32_t                                repeat
    );

protected:
    std::vector<Layer*>         layers;
};

#include "MFNN.cpp"
#endif