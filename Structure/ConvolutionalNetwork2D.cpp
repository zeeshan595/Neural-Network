#include "ConvolutionalNetwork2D.h"

using namespace Core;
using namespace Structure;

ConvolutionalNetwork2D::ConvolutionalNetwork2D(int input, int hidden, int output, int amountX, int amountY, ActivationType hiddenType, ActivationType outputType, ConnectionType connection)
{
	if (input < 1 || hidden < 1 || output < 1)
	{
		std::cout << "Input, Hidden & Output cannot be lower then 1" << std::endl;
		throw 20;
	}
	if (amountX < 2 || amountY < 2)
	{
		std::cout << "Amount cannot be lower then 2 ("<< amountX << "," << amountY <<")" << std::endl;
		throw 20;
	}

	networks.resize(amountX);
	for (int i = 0; i < amountX; i++)
	{
		for (int j = 0; j < amountY; j++)
		{
			networks[i].push_back(LinearNetwork(input * 4, hidden, output, hiddenType, outputType, connection));
		}
	}

	this->amountX = amountX;
	this->amountY = amountY;
}

ConvolutionalNetwork2D::~ConvolutionalNetwork2D()
{
	for (unsigned int i = 0; i < networks.size(); i++)
	{
		networks[i].erase(networks[i].begin(), networks[i].end());
	}
	networks.erase(networks.begin(), networks.end());
}

std::vector<std::vector<std::vector<double> > > ConvolutionalNetwork2D::Compute(std::vector<std::vector<std::vector<double> > > input)
{
	if (input.size() != networks.size() + 1 || input[0].size() != networks[0].size() + 1 || input[0][0].size() != (unsigned int)networks[0][0].GetInput())
	{
		std::cout << "inputs does not match the convolutional network 2D" << std::endl;
		throw 20;
	}

	std::vector<std::vector<std::vector<double> > > result;
	result.resize(networks.size());

	for (unsigned int i = 0; i < result.size(); i++)
	{
		result[i].resize(networks[i].size());
		for (unsigned int j = 0; j < networks[i].size(); j++)
		{
			std::vector<double> left = CommonFunctions::CombineLayers(input[i][j], input[i + 1][j]);
			std::vector<double> right = CommonFunctions::CombineLayers(input[i][j + 1], input[i + 1][j + 1]);
			std::vector<double> cLayer = CommonFunctions::CombineLayers(left, right);
			result[i][j] = networks[i][j].Compute(cLayer);
		}
	}

	return result;
}