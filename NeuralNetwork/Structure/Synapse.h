#ifndef _SYNAPSE
#define _SYNAPSE

class Neuron;
class Synapse
{
public:
    Synapse();
    Synapse(Neuron* n);

    double      GetWeight();
    double      GetWeightDelta();
    Neuron*     GetConnectedFromNeuron();
    Neuron*     GetConnectedToNeuron();

    void SetWeight(double v);
    void SetWeightDelta(double d);
    //This is the neuron this synapse is coming out from.
    void SetConnectedFromNeuron(Neuron* n);
    void SetConnectedToNeuron(Neuron* n);

private:
    double              synapse_weight          = 0;
    double              weight_delta            = 0;
    Neuron*             connected_from_neuron   = NULL;
    Neuron*             connected_to_neuron     = NULL;
};

#include "Synapse.cpp"
#endif