MFNN::MFNN(
    std::vector<uint32_t>   neurons_per_layer
){
    //Make sure layers array is valid
    if (neurons_per_layer.size() < 2)
    {
        throw std::runtime_error("ERROR [MFNN]: layers must have a size greater then or equal to 2.");
    }

    layers.resize(neurons_per_layer.size());
    for (uint32_t i = 0; i < neurons_per_layer.size(); i++)
    {
        //Make sure there is no layer with 0 nodes
        if (neurons_per_layer[i] == 0)
        {
            std::stringstream layer_id; layer_id << i;
            throw std::runtime_error("ERROR [MFNN]: layer " + layer_id.str() + " has 0 nodes inside it. A layer cannot have 0 nodes.");
        }

        //Setup Neurons
        layers[i] = new Layer();
        layers[i]->neurons.resize(neurons_per_layer[i]);
        for (uint32_t j = 0; j < neurons_per_layer[i]; j++)
        {
            layers[i]->neurons[j] = new Neuron();
        }
        //Setup Synapsis
        if (i + 1 < neurons_per_layer.size())
        {
            layers[i]->synapsis.resize(neurons_per_layer[i] * neurons_per_layer[i + 1]);
            for (uint32_t j = 0; j < neurons_per_layer[i + 1]; j++)
            {
                for (uint32_t k = 0; k < neurons_per_layer[i]; k++)
                {
                    uint32_t convert_id = (j * neurons_per_layer[i]) + k;
                    layers[i]->synapsis[convert_id] = new Synapse();
                }
            }
        }
    }

    //Connect Neurons & Synapsis
    for (uint32_t i = 0; i < neurons_per_layer.size() - 1; i++)
    {
        for (uint32_t j = 0; j < neurons_per_layer[i + 1]; j++)
        {
            for (uint32_t k = 0; k < neurons_per_layer[i]; k++)
            {
                uint32_t convert_id = (j * neurons_per_layer[i]) + k;
                layers[i]->synapsis[convert_id]->SetConnectedNeuron(layers[i]->neurons[k]);
                layers[i+1]->neurons[j]->AddConnectedSynapse(layers[i]->synapsis[convert_id]);
            }
        }
    }
}

MFNN::~MFNN()
{
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
            delete layers[i]->neurons[j];

        for (uint32_t j = 0; j < layers[i]->synapsis.size(); j++)
            delete layers[i]->synapsis[j];

        delete layers[i];
    }
}

std::vector<double> MFNN::Compute(
    std::vector<double>     inputs
){
    //Insure inputs provided are valid
    if (inputs.size() != layers[0]->neurons.size())
    {
        throw std::runtime_error("ERROR [Compute]: Input layer does not match input data provided.");
    }
    //Stup Input
    for (uint32_t i = 0; i < layers[0]->neurons.size(); i++)
    {
        layers[0]->neurons[i]->SetValue(inputs[i]);
    }
    //Compute Values
    for (uint32_t i = 1; i < layers.size(); i++)
    {
        for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
        {
            layers[i]->neurons[j]->ComputeValue();
        }
    }

    return GetOutput();
}

double MFNN::ComputeMeanSquaredError(
    std::vector<std::vector<double> >   data,
    std::vector<double>                 weights
){
    SetWeights(weights);

    uint32_t    input_layer_size    = layers[0]->neurons.size();
    uint32_t    output_layer_size   = layers[layers.size() - 1]->neurons.size();
    std::vector<double> xValues(input_layer_size); // Inputs
	std::vector<double> tValues(output_layer_size); //Outputs

	double sum_squared_error = 0.0;
	for (unsigned int i = 0; i < data.size(); ++i)
	{
		// assumes data has x-values followed by y-values
		std::copy(data[i].begin(), data[i].begin() + input_layer_size, xValues.begin());
		std::copy(data[i].begin() + input_layer_size, data[i].begin() + input_layer_size + output_layer_size, tValues.begin());

		std::vector<double> yValues = Compute(xValues);
		for (unsigned int j = 0; j < yValues.size(); ++j)
			sum_squared_error += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
	}

	return sum_squared_error;
}

void MFNN::SetWeights(
    std::vector<double>     weights
){
    uint32_t k = 0;
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
        {
            layers[i]->neurons[j]->SetBias(weights[k]);
            k++;
        }
        for (uint32_t j = 0; j < layers[i]->synapsis.size(); j++)
        {
            layers[i]->synapsis[j]->SetWeight(weights[k]);
            k++;
        }
    }
}
std::vector<double> MFNN::GetWeights()
{
    std::vector<double> result;
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
        {
            result.push_back(layers[i]->neurons[j]->GetBias());
        }
        for (uint32_t j = 0; j < layers[i]->synapsis.size(); j++)
        {
            result.push_back(layers[i]->synapsis[j]->GetWeight());
        }
    }

    return result;
}


MFNN::Layer* MFNN::GetLayer(
    uint32_t        layer_number
){
    return layers[layer_number];
}

std::vector<double> MFNN::GetOutput()
{
    std::vector<double> result;
    uint32_t output_layer       = layers.size() - 1;
    uint32_t output_layer_size  = layers[output_layer]->neurons.size();
    result.resize(output_layer_size);
    for (uint32_t i = 0; i < output_layer_size; i++)
    {
        result[i] = layers[output_layer]->neurons[i]->GetValue();
    }

    return result;
}

void MFNN::SetRandomWeights()
{
    std::srand(std::time(NULL));
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        for (uint32_t j = 0; j < layers[i]->synapsis.size(); j++)
        {
            double rnd = (double)std::rand() / (double)RAND_MAX;
            layers[i]->synapsis[j]->SetWeight(rnd);
        }
    }
}