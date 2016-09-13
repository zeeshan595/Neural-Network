#ifndef _ACTIVATION
#define _ACTIVATION

#include <math.h>
#include <vector>

namespace Core
{
	enum ActivationType
	{
		NONE,
		LOGISTIC_SIGMOID,
		HYPERBOLIC_TANGENT,
		HEAVISIDE_STEP,
		SOFTMAX
	};

	class ActivationMethods
	{
	public:
		static std::vector<double> Activation(std::vector<double> xValues, ActivationType type);

	private:
		static double LogisticSigmoid(double values);
		static double HyperbolicTangent(double values);
		static double HeavisideStep(double values);
		static std::vector<double> Softmax(std::vector<double> values);
	};
};

#endif