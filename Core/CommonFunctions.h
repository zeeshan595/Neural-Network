#ifndef _COMMON_FUNCTIONS
#define _COMMON_FUNCTIONS

#include <vector>

namespace Core
{
	class CommonFunctions
	{
	public:
		static std::vector<double> CombineLayers(std::vector<double> layer1, std::vector<double> layer2);
		//static std::pair<double[],double[]> DisjoinLayers(double[] layer, int layer1Length);
	};
};

#endif