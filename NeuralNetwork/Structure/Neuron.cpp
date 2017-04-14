double                          Neuron::GetValue()
{
    return neuron_value;
}
double                          Neuron::GetGradient()
{
    return neuron_gradient;
}
double                          Neuron::GetBias()
{
    return bias_value;
}
double                          Neuron::GetBiasDelta()
{
    return bias_delta;
}
Activation::ActivationType      Neuron::GetActivationType()
{
    return activation_type;
}
std::vector<Synapse*>*          Neuron::GetConnectedSynapsisIn()
{
    return &synapsis_in;
}

std::vector<Synapse*>*          Neuron::GetConnectedSynapsisOut()
{
    return &synapsis_out;
}

void Neuron::ComputeValue()
{
    //Sets the current value to zero.
    neuron_value = 0;
    //Goes through each synapse connected to this neuron that
    //is feeding into this neuron.
    for (uint32_t i = 0; i < synapsis_in.size(); i++)
    {
        //The weight of the synapse and the value of the node that synapse
        //is coming out from are multiplied then added to the value of this
        //neuron
        neuron_value += synapsis_in[i]->GetWeight() * synapsis_in[i]->GetConnectedFromNeuron()->GetValue();
    }
    //The bias value is added to the neuron value.
    neuron_value += bias_value;
    //An activation method is applied to this neuron
    neuron_value = Activation::ApplyFunction({ neuron_value }, activation_type)[0];
}
void Neuron::SetValue(double v)
{
    neuron_value = v;
}
void Neuron::SetGradient(double g)
{
    neuron_gradient = g;
}
void Neuron::SetBias(double b)
{
    bias_value = b;
}
void Neuron::SetBiasDelta(double d)
{
    bias_delta = d;
}
void Neuron::SetActivationType(Activation::ActivationType type)
{
    activation_type = type;
}


void Neuron::SetConnectedSynapsisIn(std::vector<Synapse*> synapsis)
{
    synapsis_in = synapsis;
}
void Neuron::AddConnectedSynapseIn(Synapse* synapse)
{
    synapsis_in.push_back(synapse);
}

void Neuron::SetConnectedSynapsisOut(std::vector<Synapse*> synapsis)
{
    synapsis_out = synapsis;
}
void Neuron::AddConnectedSynapseOut(Synapse* synapse)
{
    synapsis_out.push_back(synapse);
}