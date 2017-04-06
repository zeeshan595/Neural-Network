#ifndef _ACTIVATION
#define _ACTIVATION

namespace Activation
{
    enum ActivationType{
        LOGISTIC_SIGMOID,
        HYPERBOLIC_TANGENT,
        HEAVISIDE_STEP,
        SOFTMAX
    };

    //Applies a particular type of activation function
    //to a list of numbers(double)
    std::vector<double>     ApplyFunction(
        std::vector<double>     v,
        ActivationType          type
    );

    //Applies a particular type of activation function's
    //derivative to a list of numbers(double). 
    //NOTE: HEAVISIDE_STEP's derivative cannot be computed.
    std::vector<double>     ApplyDerivative(
        std::vector<double>     v,
        ActivationType          type
    );

    //Activation Functions
    std::vector<double>     LogisticSigmoid(
        std::vector<double>     v
    );
    std::vector<double>     HyperbolicTangent(
        std::vector<double>     v
    );
    std::vector<double>     HeavisideStep(
        std::vector<double>     v
    );
    std::vector<double>     Softmax(
        std::vector<double>     v
    );

    //Inverse Functions
    std::vector<double>     LogisticSigmoidDerivative(
        std::vector<double>     v
    );
    std::vector<double>     HyperbolicTangentDerivative(
        std::vector<double>     v
    );
};

#include "Activation.cpp"
#endif