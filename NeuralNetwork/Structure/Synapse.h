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
    //Stores the current weight of this synapse
    double              synapse_weight          = 0;
    //Stores the derivative error of this synapse
    double              weight_delta            = 0;
    //Stores a pointer a neuron to which this neuron is coming out from.
    Neuron*             connected_from_neuron   = NULL;
    //Stores a pointer a neuron to which this synapse is going into
    Neuron*             connected_to_neuron     = NULL;
};

#include "Synapse.cpp"
#endif