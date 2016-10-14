#include "Activation.h"

#include <cassert>
#include <math.h>

std::vector<double> Core::Activation::ApplyActivation(std::vector<double> xValues, ActivationType type)
{
    std::vector<double> result(xValues.size());
    if (type == LOGISTIC_SIGMOID)
    {
        for (unsigned int i = 0; i < result.size(); i++)
            result[i] = LogisticSigmoid(xValues[i]);
    }
    else if (type == HYPERBOLIC_TANGENT)
    {
        for (unsigned int i = 0; i < result.size(); i++)
            result[i] = HyperbolicTangentDerivative(xValues[i]);
    }
    else if (type == HEAVISIDE_STEP)
    {
        for (unsigned int i = 0; i < result.size(); i++)
            result[i] = HeavisideStep(xValues[i]);
    }
    else if (type == SOFTMAX)
    {
        result = Softmax(xValues);
    }
    else
    {
        for (unsigned int i = 0; i < result.size(); i++)
            result[i] = xValues[i];
    }

    return result;
}

std::vector<double> Core::Activation::InverseActivation(std::vector<double> xValues, ActivationType type)
{
    assert(type != HEAVISIDE_STEP);

    std::vector<double> result(xValues.size());
    if (type == LOGISTIC_SIGMOID)
    {
        for (unsigned int i = 0; i < result.size(); i++)
            result[i] = LogisticSigmoidDerivative(xValues[i]);
    }
    else if (type == HYPERBOLIC_TANGENT)
    {
        for (unsigned int i = 0; i < result.size(); i++)
            result[i] = HyperbolicTangent(xValues[i]);
    }
    else if (type == SOFTMAX)
    {
        for (unsigned int i = 0; i < result.size(); i++)
            result[i] = LogisticSigmoidDerivative(xValues[i]);
    }
    else
    {
        for (unsigned int i = 0; i < result.size(); i++)
            result[i] = xValues[i];
    }

    return result;
}

double Core::Activation::LogisticSigmoid(double values)
{
    return 1.0 / (1.0 + exp(-values));
}

double Core::Activation::HyperbolicTangent(double values)
{
    return tanh(values);
}

double Core::Activation::HeavisideStep(double values)
{
    if (values < 0)
        return 0;
    else
        return 1;
}

std::vector<double> Core::Activation::Softmax(std::vector<double> values)
{
    // determine max output sum
	// does all output nodes at once so scale doesn't have to be re-computed each time
	double max = values[0];
	for (unsigned int i = 0; i < values.size(); i++)
		if (values[i] > max)
			max = values[i];
	
	//determine scaling factor -- sum of exp (each val - max)
	double scale = 0.0;
	for (unsigned int i = 0; i < values.size(); i++)
		scale += exp(values[i] - max);
	
	std::vector<double> result;
	result.resize(values.size());
	for (unsigned int i = 0; i < values.size(); i++)
		result[i] = exp(values[i] - max) / scale;

	return result;
}

/*INVERSE*/

double Core::Activation::LogisticSigmoidDerivative(double values)
{
    return LogisticSigmoid(values) * (1 - LogisticSigmoid(values));
}

double Core::Activation::HyperbolicTangentDerivative(double values)
{
    return 1 - pow(HyperbolicTangent(values), 2);
}