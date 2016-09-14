#include "CommonFunctions.h"

using namespace Core;

std::vector<double> CommonFunctions::CombineLayers(std::vector<double> layer1, std::vector<double> layer2)
{
	std::vector<double> result;
	result.resize(layer1.size() + layer2.size());
	int k = 0;
	for (unsigned int i = 0; i < layer1.size(); i++)
	{
		result[k] = layer1[i];
		k++;
	}

	for (unsigned int i = 0; i < layer2.size(); i++)
	{
		result[k] = layer2[i];
		k++;
	}

	return result;
}
/*
std::pair<std::vector<double>, std::vector<double> > CommonFunctions::DisjoinLayers(std::vector<double> layer, int layerLength)
{
	return nullptr;
}
*/