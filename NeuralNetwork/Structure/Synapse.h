#ifndef _SYNAPSE
#define _SYNAPSE

class Neuron;
class Synapse
{
public:
    Synapse();
    Synapse(Neuron* n);

    double      GetWeight();
    Neuron*     GetConnectedNeuron();

    void SetWeight(double v);
    //This is the neuron this synapse is coming out from.
    void SetConnectedNeuron(Neuron* n);

private:
    double              synapse_weight          = 0;
    Neuron*             connected_neuron        = NULL;
};

#include "Synapse.cpp"
#endif