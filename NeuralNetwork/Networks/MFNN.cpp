MFNN::MFNN(
    std::vector<uint32_t>   neurons_per_layer
){
    //Make sure layers array is valid
    if (neurons_per_layer.size() < 2)
    {
        throw std::runtime_error("ERROR [MFNN]: layers must have a size greater then or equal to 2.");
    }

    layers.resize(neurons_per_layer.size());
    //Goes through each layer and the neurons/synapsis inside the layer then initializes them by calling the constructor.
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
                //This is a formula used to order the synapsis so a specfic synapse can be
                //extracted later on in the process.
                uint32_t convert_id = (j * neurons_per_layer[i]) + k;
                //Set synapsis pointers and neuron pointers to correct values
                //making sure they can access each others values and are connected.
                layers[i]->synapsis[convert_id]->SetConnectedToNeuron(layers[i+1]->neurons[j]);
                layers[i]->synapsis[convert_id]->SetConnectedFromNeuron(layers[i]->neurons[k]);
                
                layers[i]->neurons[k]->AddConnectedSynapseOut(layers[i]->synapsis[convert_id]);
                layers[i+1]->neurons[j]->AddConnectedSynapseIn(layers[i]->synapsis[convert_id]);
            }
        }
    }

    SetRandomWeights();
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
    //Stup Input by assigning the values provided to the first layer
    //of MFNN.
    for (uint32_t i = 0; i < layers[0]->neurons.size(); i++)
    {
        layers[0]->neurons[i]->SetValue(inputs[i]);
    }
    //Compute Values by going through each neuron and calling the compute function
    for (uint32_t i = 1; i < layers.size(); i++)
    {
        for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
        {
            layers[i]->neurons[j]->ComputeValue();
        }
    }

    //output the last layer in the MFNN
    return GetOutput();
}

double MFNN::GetMeanSquaredError(
    std::vector<std::vector<double> >   data,
    std::vector<double>                 weights
){
    //Error Checking
    if (data.size() <= 0)
        throw std::runtime_error("ERROR [GetMeanSquaredError]: could not locate data.");

    //Get the size of input & output layer to ensure data matches it.
    uint32_t    input_layer_size    = layers[0]->neurons.size();
    uint32_t    output_layer_size   = layers[layers.size() - 1]->neurons.size();
    if (data[0].size() != input_layer_size + output_layer_size)
        throw std::runtime_error("ERROR [GetMeanSquaredError]: trainning data does not match neural network");

    //Set the current weights to equal to the weights provided through
    //the parameter
    SetWeights(weights);

    //Setup input and desired output variables.
    std::vector<double> xValues(input_layer_size); // Inputs
	std::vector<double> tValues(output_layer_size); //Outputs

    //Go through each training data sent from the "data" parameter
    //Set sum squared error to 0
	double sum_squared_error = 0.0;
	for (uint32_t i = 0; i < data.size(); ++i)
	{
		//Extract data from "data" parameter and store it in the xValues/tValues variables.
		std::copy(data[i].begin(), data[i].begin() + input_layer_size, xValues.begin());
		std::copy(data[i].begin() + input_layer_size, data[i].begin() + input_layer_size + output_layer_size, tValues.begin());

        //Get the output value computed by the neural network
		std::vector<double> yValues = Compute(xValues);
        //Go through each node and and add the (computed value - desired value)^2
        //to the sum squared error
		for (uint32_t j = 0; j < yValues.size(); ++j)
			sum_squared_error += ((yValues[j] - tValues[j]) * (yValues[j] - tValues[j]));
	}

    //Return the sum squared error
	return sum_squared_error;
}
double MFNN::GetAccuracy(
    std::vector<std::vector<double> > data
){
    //create 2 variables that will be used to determain if the value
    //computed is accurate.
    uint32_t    correct             = 0;
    uint32_t    wrong               = 0;
    //Get the size of output & input layers and store them in a variable
    //for ease of access
    uint32_t    input_layer_size    = layers[0]->neurons.size();
    uint32_t    output_layer_size   = layers[layers.size() - 1]->neurons.size();
    //Setup variables that will store the input, computed output and desired output
    std::vector<double> xValues(input_layer_size); // Inputs
	std::vector<double> tValues(output_layer_size); //Outputs
    std::vector<double> yValues(output_layer_size);

    //Loop through all the data provided
    for (uint32_t i = 0; i < data.size(); ++i)
	{
        //Fill the input, desired output and then compute the actual output
        std::copy(data[i].begin(), data[i].begin() + input_layer_size, xValues.begin());
		std::copy(data[i].begin() + input_layer_size, data[i].begin() + input_layer_size + output_layer_size, tValues.begin());
        yValues = Compute(xValues);

        //Check the maximum value of each output node. If the data provided
        //Also has the maximum value in that place then the neural network
        //predicted it correctly, otherwise the prediction was wrong.
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
        //Add to the list if it is correct or wrong.
        if (max_computed == max_target)
            correct++;
        else
            wrong++;
    }

    //Return accuracy by deviding the correct amount by total data.
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
    std::vector<double> new_weights;
    double high = +0.1;
    double low  = -0.1;
    for (uint32_t i = 0; i < layers.size(); i++)
    {
        for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
        {
            double r = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
            new_weights.push_back(r);
        }
        for (uint32_t j = 0; j < layers[i]->synapsis.size(); j++)
        {
            double r = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
            new_weights.push_back(r);
        }
    }
    SetWeights(new_weights);
}

void MFNN::TrainUsingBP(
    //Training data
    std::vector<std::vector<double> >       train_data,
    //learning rate free parameter
    double                                  learning_rate,
    //momentum free parameter
    double                                  momentum,
    //weight decay free parameter
    double                                  weight_decay,
    //The max epochs this should run for.
    uint32_t                                repeat
){
    //Error Checking
    //Make sures the training data does not equal NULL
    if (train_data.size() <= 0)
        throw std::runtime_error("ERROR [TrainUsingBP]: Train data is does not contain any data");
    //Make sures the training data matches the neural network that is being trained 
    if (train_data[0].size() != layers[0]->neurons.size() + layers[layers.size() - 1]->neurons.size())
        throw std::runtime_error("ERROR [TrainUsingBP]: train data size does not match the neural network");
    //max epochs cannot be smaller then 1
    if (repeat < 1)
        throw std::runtime_error("ERROR [TrainUsingBP]: Repeat must be greater than 0");

    //Setup
    uint32_t                                repeat_counter                      = 0;
    uint32_t                                input_layer_size                    = layers[0]->neurons.size();
    uint32_t                                output_layer_size                   = layers[layers.size() - 1]->neurons.size();

    while (repeat_counter < repeat)
    {
        //stores the current time in "begin" variable. This time is highly accurate.
        auto begin = std::chrono::high_resolution_clock::now();
        for (uint32_t d = 0; d < train_data.size(); d++)
        {
            //Get inputs, desired outputs and computed outputs so the training process can begin.
            std::vector<double> xValues(input_layer_size); // Inputs
            std::vector<double> tValues(output_layer_size); //Outputs
            std::copy(train_data[d].begin(), train_data[d].begin() + input_layer_size, xValues.begin());
            std::copy(train_data[d].begin() + input_layer_size, train_data[d].begin() + input_layer_size + output_layer_size, tValues.begin());
            std::vector<double> yValues = Compute(xValues);

            //Compute gradiant for each neuron from output layer to input layer.
            for (uint32_t i = layers.size() - 1; i > 0; i--)
            {
                for (uint32_t j = 0; j < layers[i]->neurons.size(); j++)
                {
                    //Apply the derivative of the activation function
                    double derivative   = Activation::ApplyDerivative({ layers[i]->neurons[j]->GetValue() }, layers[i]->neurons[j]->GetActivationType())[0];
                    double sum          = 0.0;
                    if (i == layers.size() - 1) //If Output layer
                    {
                        //If it is the output layer the derivative error is desired - actual output
                        sum = (tValues[j] - yValues[j]);
                    }
                    else //Else other layer
                    {
                        //Go through each synapse going out from this neuron
                        for (uint32_t k = 0; k < layers[i + 1]->neurons.size(); k++)
                        {
                            //Multiply the synapse weight by the node gradient to which this synapse is going to.
                            uint32_t convert_id = (k * layers[i]->neurons.size()) + j;
                            double x = layers[i + 1]->neurons[k]->GetGradient() * layers[i]->synapsis[convert_id]->GetWeight();
                            //The resulting value is added to a sum
                            sum += x;
                        }
                    }
                    //The gradient of the node is then derivative multiplied by the sum.
                    layers[i]->neurons[j]->SetGradient(derivative * sum);
                }
            }

            //Update Weights for each synapse
            for (uint32_t i = 0; i < layers.size() - 1; i++)
            {
                for (uint32_t j = 0; j < layers[i]->synapsis.size(); j++)
                {
                    //get delta by multiply learning rate (free parameter) with synapse gradient
                    //and then multiplying that by neuron value.
                    double delta    = learning_rate * layers[i]->synapsis[j]->GetConnectedToNeuron()->GetGradient() * layers[i]->synapsis[j]->GetConnectedFromNeuron()->GetValue();
                    //Add the delta to the current synapse weight
                    double weight   = layers[i]->synapsis[j]->GetWeight();
                    weight         += delta;
                    //add momentum multiplied by previous delta to the current weight.
                    weight         += momentum * layers[i]->synapsis[j]->GetWeightDelta();
                    //multiply weight decay by current weight and then subtract the result from
                    //current weight
                    weight         -= (weight_decay * weight);
                    //Update all values
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

        //get current time and subtract it from the time noted in "begin" variable
        auto elapsed_secs = std::chrono::high_resolution_clock::now() - begin;
        //Convert time to microseconds
        long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_secs).count();
        //Output text to the console.
        std::cout << "epoch: " << repeat_counter << ", accuracy:" << GetAccuracy(train_data) << std::endl;
        //Go to next epoch
        repeat_counter++;
    }
}