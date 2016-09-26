#ifndef _ACTIVATION
#define _ACTIVATION

#include "ActivationType.h"

#include <vector>

namespace Core
{
    class Activation
    {
    public:
        static std::vector<double> ApplyActivation(std::vector<double> xValues, ActivationType type);

    private:
        static double LogisticSigmoid(double values);
        static double HyperbolicTangent(double values);
        static double HeavisideStep(double values);
        static std::vector<double> Softmax(std::vector<double> values);
    };
};

#endif