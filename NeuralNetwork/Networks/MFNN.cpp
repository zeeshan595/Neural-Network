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
                layers[i]->synapsis[convert_id]->SetConnectedToNeuron(layers[i+1]->neurons[j]);
                layers[i]->synapsis[convert_id]->SetConnectedFromNeuron(layers[i]->neurons[k]);
                
                layers[i]->neurons[k]->AddConnectedSynapseOut(layers[i]->synapsis[convert_id]);
                layers[i+1]->neurons[j]->AddConnectedSynapseIn(layers[i]->synapsis[convert_id]);
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

double MFNN::GetMeanSquaredError(
    std::vector<std::vector<double> >   data,
    std::vector<double>                 weights
){
    //Error Checking
    if (data.size() <= 0)
        throw std::runtime_error("ERROR [GetMeanSquaredError]: could not locate data.");

    uint32_t    input_layer_size    = layers[0]->neurons.size();
    uint32_t    output_layer_size   = layers[layers.size() - 1]->neurons.size();
    if (data[0].size() != input_layer_size + output_layer_size)
        throw std::runtime_error("ERROR [GetMeanSquaredError]: trainning data does not match neural network");

    SetWeights(weights);

    std::vector<double> xValues(input_layer_size); // Inputs
	std::vector<double> tValues(output_layer_size); //Outputs

	double sum_squared_error = 0.0;
	for (uint32_t i = 0; i < data.size(); ++i)
	{
		// assumes data has x-values followed by y-values
		std::copy(data[i].begin(), data[i].begin() + input_layer_size, xValues.begin());
		std::copy(data[i].begin() + input_layer_size, data[i].begin() + input_layer_size + output_layer_size, tValues.begin());

		std::vector<double> yValues = Compute(xValues);
		for (uint32_t j = 0; j < yValues.size(); ++j)
			sum_squared_error += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
	}

	return sum_squared_error;
}
double MFNN::GetAccuracy(
    std::vector<std::vector<double> > data
){
    uint32_t    correct             = 0;
    uint32_t    wrong               = 0;
    uint32_t    input_layer_size    = layers[0]->neurons.size();
    uint32_t    output_layer_size   = layers[layers.size() - 1]->neurons.size();
    std::vector<double> xValues(input_layer_size); // Inputs
	std::vector<double> tValues(output_layer_size); //Outputs
    std::vector<double> yValues(output_layer_size);

    for (uint32_t i = 0; i < data.size(); ++i)
	{
        std::copy(data[i].begin(), data[i].begin() + input_layer_size, xValues.begin());
		std::copy(data[i].begin() + input_layer_size, data[i].begin() + input_layer_size + output_layer_size, tValues.begin());
        yValues = Compute(xValues);

        uint32_t max_computed   = 0;
        uint32_t max_target     = 0;
        for (uint32_t j = 0; j < output_layer_size; j++)
        {
            if (yValues[max_computed] < yValues[j])
                max_computed = j;
        }
        for (uint32_t j = 0; j < output_layer_size; j++)
        {
            if (tValues[max_target] < tValues[j])
                max_target = j;
        }
        if (max_computed == max_target)
            correct++;
        else
            wrong++;
    }

    return (double)correct / (double)(correct + wrong);
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

void MFNN::TrainUsingBP(
    std::vector<std::vector<double> >       train_data,
    double                                  learning_rate,
    double                                  momentum,
    double                                  weight_decay,
    uint32_t                                repeat
){
    //Error Checking
    if (train_data.size() <= 0)
        throw std::runtime_error("ERROR [BackPropagationTrain]: Train data is does not contain any data");
    if (train_data[0].size() != layers[0]->neurons.size() + layers[layers.size() - 1]->neurons.size())
        throw std::runtime_error("ERROR [BackPropagationTrain]: train data size does not match the neural network");
    if (momentum <= 0)
        throw std::runtime_error("ERROR [BackPropagationTrain]: momentum must be greater then 0");
    
    //Setup
    uint32_t                                repeat_counter                      = 0;
    uint32_t                                input_layer_size                    = layers[0]->neurons.size();
    uint32_t                                output_layer_size                   = layers[layers.size() - 1]->neurons.size();

    while (repeat_counter < repeat)
    {
        //Used to measure the duration per epoch
        auto begin = std::chrono::high_resolution_clock::now();
        for (uint32_t d = 0; d < train_data.size(); d++)
        {
            std::vector<double> xValues(input_layer_size); // Inputs
            std::vector<double> tValues(output_layer_size); //Outputs
            std::copy(train_data[d].begin(), train_data[d].begin() + input_layer_size, xValues.begin());
            std::copy(train_data[d].begin() + input_layer_size, train_data[d].begin() + input_layer_size + output_layer_size, tValues.begin());
            std::vector<double> yValues = Compute(xValues);

            //Compute gradiant for each neuron
            for (uint32_t i = layers.size() - 1; i > 0; i--)
            {
                for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
                {
                    double derivative   = Activation::ApplyDerivative({ layers[i]->neurons[j]->GetValue() }, layers[i]->neurons[j]->GetActivationType())[0];
                    double sum          = 0.0;
                    if (i == layers.size() - 1) //If Output layer
                    {
                        sum = (tValues[j] - yValues[j]);
                    }
                    else //Else other layer
                    {
                        for (uint32_t k = 0; k < layers[i + 1]->neurons.size(); k++)
                        {
                            uint32_t convert_id = (k * layers[i]->neurons.size()) + j;
                            double x = layers[i + 1]->neurons[k]->GetGradient() * layers[i]->synapsis[convert_id]->GetWeight();
                            sum += x;
                        }
                    }
                    layers[i]->neurons[j]->SetGradient(derivative * sum);
                }
            }

            //Update Weights
            for (uint32_t i = 0; i < layers.size() - 1; i++)
            {
                for (uint32_t j = 0; j < layers[i]->synapsis.size(); j++)
                {
                    double delta    = learning_rate * layers[i]->synapsis[j]->GetConnectedToNeuron()->GetGradient() * layers[i]->synapsis[j]->GetConnectedFromNeuron()->GetValue();
                    double weight   = layers[i]->synapsis[j]->GetWeight();
                    weight         += delta;
                    weight         += momentum * layers[i]->synapsis[j]->GetWeightDelta();
                    weight         -= (weight_decay * weight);
                    layers[i]->synapsis[j]->SetWeight(weight);
                    layers[i]->synapsis[j]->SetWeightDelta(delta);
                }
            }

            //Update Biases
            for (uint32_t i = 1; i < layers.size(); i++)
            {
                for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
                {
                    double delta    = learning_rate * layers[i]->neurons[j]->GetGradient() * 1.0;
                    double bias     = layers[i]->neurons[j]->GetBias();
                    bias           += delta;
                    bias           += momentum * layers[i]->neurons[j]->GetBiasDelta();
                    bias           -= (weight_decay * bias);
                    layers[i]->neurons[j]->SetBias(bias);
                    layers[i]->neurons[j]->SetBiasDelta(delta);
                }
            }
        }

        //Used to measure the duration per epoch
        auto elapsed_secs = std::chrono::high_resolution_clock::now() - begin;
        long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_secs).count();
        std::cout << repeat_counter << " " << GetMeanSquaredError(train_data, GetWeights()) << " " << microseconds << std::endl;
        repeat_counter++;
    }
}