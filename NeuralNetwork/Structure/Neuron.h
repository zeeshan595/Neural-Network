#ifndef _NEURON
#define _NEURON

class Neuron
{
public:
    double                          GetValue();
    double                          GetGradient();
    double                          GetBias();
    double                          GetBiasDelta();
    Activation::ActivationType      GetActivationType();

    //Gets synapse going into this neuron or coming out of this neuron
    std::vector<Synapse*>*          GetConnectedSynapsisIn();
    std::vector<Synapse*>*          GetConnectedSynapsisOut();

    void ComputeValue();
    void SetValue(double v);
    void SetGradient(double g);
    void SetBias(double b);
    void SetBiasDelta(double d);
    void SetActivationType(Activation::ActivationType type);

    void SetConnectedSynapsisIn(std::vector<Synapse*> synapsis);
    void AddConnectedSynapseIn(Synapse* synapse);

    void SetConnectedSynapsisOut(std::vector<Synapse*> synapsis);
    void AddConnectedSynapseOut(Synapse* synapse);

private:
    double                              neuron_value            = 0;
    double                              neuron_gradient         = 0;
    double                              bias_value              = 0;
    double                              bias_delta              = 0;
    Activation::ActivationType          activation_type         = Activation::ActivationType::LOGISTIC_SIGMOID;
    std::vector<Synapse*>               synapsis_in;
    std::vector<Synapse*>               synapsis_out;
};

#include "Neuron.cpp"
#endif