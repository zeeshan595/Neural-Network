#ifndef _NEURON
#define _NEURON

class Neuron
{
public:
    double                          GetValue();
    double                          GetBias();
    Activation::ActivationType      GetActivationType();
    //Gets synapse going into this neuron
    std::vector<Synapse*>*          GetConnectedSynapsis();

    void ComputeValue();
    void SetValue(float v);
    void SetBias(float b);
    void SetActivationType(Activation::ActivationType type);
    void SetConnectedSynapsis(std::vector<Synapse*> synapsis);
    void AddConnectedSynapse(Synapse* synapse);

private:
    double                              neuron_value            = 0;
    double                              bias_value              = 0;
    Activation::ActivationType          activation_type         = Activation::ActivationType::LOGISTIC_SIGMOID;
    std::vector<Synapse*>               attached_synapsis;
};

#include "Neuron.cpp"
#endif