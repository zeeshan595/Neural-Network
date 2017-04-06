double                          Neuron::GetValue()
{
    return neuron_value;
}
double                          Neuron::GetBias()
{
    return bias_value;
}
Activation::ActivationType      Neuron::GetActivationType()
{
    return activation_type;
}
std::vector<Synapse*>*          Neuron::GetConnectedSynapsis()
{
    return &attached_synapsis;
}

void Neuron::ComputeValue()
{
    neuron_value = 0;
    for (uint32_t i = 0; i < attached_synapsis.size(); i++)
    {
        neuron_value += attached_synapsis[i]->GetWeight() * attached_synapsis[i]->GetConnectedNeuron()->GetValue();
    }
    neuron_value += bias_value;
    neuron_value = Activation::ApplyFunction({ neuron_value }, activation_type)[0];
}
void Neuron::SetValue(float v)
{
    neuron_value = v;
}
void Neuron::SetBias(float b)
{
    bias_value = b;
}
void Neuron::SetActivationType(Activation::ActivationType type)
{
    activation_type = type;
}
void Neuron::SetConnectedSynapsis(std::vector<Synapse*> synapsis)
{
    attached_synapsis = synapsis;
}
void Neuron::AddConnectedSynapse(Synapse* synapse)
{
    attached_synapsis.push_back(synapse);
}