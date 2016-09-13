#ifndef _COMMON_FUNCTIONS
#define _COMMON_FUNCTIONS

#include <iostream>

namespace Core
{
	class CommonFunctions
	{
	public:
		static double[] CombineLayers(double[] layer1, double[] layer2);
		static std::pair<double[],double[]> DisjoinLayers(double[] layer, int layer1Length);
	};
};

#endif