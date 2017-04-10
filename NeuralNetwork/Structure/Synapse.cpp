Synapse::Synapse()
{

}

double      Synapse::GetWeight()
{
    return synapse_weight;
}
double      Synapse::GetWeightDelta()
{
    return weight_delta;
}
Neuron*     Synapse::GetConnectedFromNeuron()
{
    return connected_from_neuron;
}
Neuron*     Synapse::GetConnectedToNeuron()
{
    return connected_to_neuron;
}

void Synapse::SetWeight(double v)
{
    synapse_weight = v;
}
void Synapse::SetWeightDelta(double d)
{
    weight_delta = d;
}
void Synapse::SetConnectedFromNeuron(Neuron* n)
{
    connected_from_neuron = n;
}
void Synapse::SetConnectedToNeuron(Neuron* n)
{
    connected_to_neuron = n;
}