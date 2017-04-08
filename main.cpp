#include "NeuralNetwork/Common.h"
MFNN nn = MFNN({ 4, 7, 3 });

std::vector<double> compute_func(std::vector<double> input)
{
    return nn.Compute(input);
}

double error_compute_func(std::vector<std::vector<double> > data, std::vector<double> weights)
{
    return nn.ComputeMeanSquaredError(data, weights);
}

int main()
{
    std::cout << "Hello World" << std::endl;

    nn.SetWeights(IrisData::GOOD_WEIGHTS);
    //std::vector<double> new_weights = PSO::Train(IrisData::dataset, 20, 0.1, 0.05, 1000, nn.GetWeights(), compute_func, error_compute_func);
    std::vector<double> output = nn.Compute({6.3,2.3,4.4,1.3});

    std::cin.ignore();
    return 0;
}