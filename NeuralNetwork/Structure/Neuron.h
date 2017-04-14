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
    //stores the current neuron value
    double                              neuron_value            = 0;
    //stores the current neuron gradient (used by BP).
    double                              neuron_gradient         = 0;
    //stores the bias value that will be applied to this neuron
    double                              bias_value              = 0;
    //stores the bias delta for this neuron (used by BP)
    double                              bias_delta              = 0;
    //Stores the activation function that will be applied in this neuron
    Activation::ActivationType          activation_type         = Activation::ActivationType::LOGISTIC_SIGMOID;
    //Stores a list of pointers. These pointers point to all the synapsis going
    //into this neuron
    std::vector<Synapse*>               synapsis_in;
    //Stores a list of pointers. These pointers point to all the synapsis coming
    //out of neuron
    std::vector<Synapse*>               synapsis_out;
};

#include "Neuron.cpp"
#endif