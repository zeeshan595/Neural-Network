std::vector<double> Activation::ApplyFunction(
    std::vector<double> v,
    ActivationType type
){
    switch(type)
    {
        case LOGISTIC_SIGMOID:
            return LogisticSigmoid(v);
        case HYPERBOLIC_TANGENT:
            return HyperbolicTangent(v);
        case HEAVISIDE_STEP:
            return HeavisideStep(v);
        case SOFTMAX:
            return Softmax(v);
        default:
            std::cout << "ERROR [ApplyFunction]: unkown activation type was used." << std::endl;
            return v;
    }
}
std::vector<double> Activation::ApplyDerivative(
    std::vector<double> v,
    ActivationType type
){
    switch(type)
    {
        case LOGISTIC_SIGMOID:
            return LogisticSigmoidDerivative(v);
        case HYPERBOLIC_TANGENT:
            return HyperbolicTangentDerivative(v);
        case HEAVISIDE_STEP:
            std::cout << "ERROR [ApplyDerivative]: cannot compute the derivative of heaviside step function" << std::endl;
            return v;
        case SOFTMAX:
            return LogisticSigmoidDerivative(v);
        default:
            std::cout << "ERROR [ApplyDerivative]: unkown activation type was used." << std::endl;
            return v;
    }
}

//Activation Functions
std::vector<double> Activation::LogisticSigmoid(
    std::vector<double> v
){
    //An array of numbers used to store results in.
    std::vector<double> result;
    //Resize the array to match the parameter supplied by the user.
    result.resize(v.size());
    //go through each number provided by the parameter
    for (uint32_t i = 0; i < v.size(); i++)
    {
        //Apply the logistic sigmoid function to each number
        //and store it in result array.
        result[i] = 1.0 / (1.0 + exp(-v[i]));
    }
    //return the result array.
    return result;
}
std::vector<double> Activation::HyperbolicTangent(
    std::vector<double> v
){
    std::vector<double> result;
    result.resize(v.size());
    for (uint32_t i = 0; i < v.size(); i++)
    {
        result[i] = tanh(v[i]);
    }
    return result;
}
std::vector<double> Activation::HeavisideStep(
    std::vector<double> v
){
    std::vector<double> result;
    result.resize(v.size());
    for (uint32_t i = 0; i < v.size(); i++)
    {
        if (v[i] < 0)
            result[i] = 0;
        else
            result[i] = 1;
    }
    return result;
}
std::vector<double> Activation::Softmax(
    std::vector<double> v
){
    //Get max value
    double max = v[0];
    for (uint32_t i = 0; i < v.size(); i++)
    {
        if (max < v[i])
        {
            max = v[i];
        }
    }

    //Determine scaling factor -- sum of exp (each val - max)
    double scale = 0.0;
    for (uint32_t i = 0; i < v.size(); i++)
    {
        scale += exp(v[i] - max);
    }

    std::vector<double> result;
	result.resize(v.size());
	for (unsigned int i = 0; i < v.size(); i++)
    {
		result[i] = exp(v[i] - max) / scale;
    }
    
	return result;
}

//Derivative Functions
std::vector<double> Activation::LogisticSigmoidDerivative(
    std::vector<double> v
){
    std::vector<double> result;
    result.resize(v.size());
    for (uint32_t i = 0; i < result.size(); i++)
    {
        result[i] = (1 - v[i]) * v[i];
    }
    return result;
}
std::vector<double> Activation::HyperbolicTangentDerivative(
    std::vector<double> v
){
    std::vector<double> result;
    result.resize(v.size());
    for (uint32_t i = 0; i < result.size(); i++)
    {
        result[i] = (1 - v[i]) * (1 + v[i]);
    }
    return result;
}