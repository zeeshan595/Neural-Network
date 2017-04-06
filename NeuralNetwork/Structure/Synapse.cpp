Synapse::Synapse()
{

}
Synapse::Synapse(Neuron* n)
{
    connected_neuron = n;
}

double      Synapse::GetWeight()
{
    return synapse_weight;
}
Neuron*     Synapse::GetConnectedNeuron()
{
    return connected_neuron;
}

void Synapse::SetWeight(double v)
{
    synapse_weight = v;
}
void Synapse::SetConnectedNeuron(Neuron* n)
{
    connected_neuron = n;
}