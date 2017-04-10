#include "NeuralNetwork/Common.h"

int main()
{
    std::cout << "Hello World" << std::endl;

    MFNN nn = MFNN({ 4, 7, 3 });

   // nn.SetWeights(IrisData::GOOD_WEIGHTS);
    //PSO::Train(IrisData::dataset, 20, 0.05, 1000, dynamic_cast<BaseNetwork*>(&nn));
    //GA::Train(IrisData::dataset, 20, 5, 1000, dynamic_cast<BaseNetwork*>(&nn));
    nn.BackPropagationTrain(IrisData::dataset, 0.05, 0.01, 0, 2000);
    std::vector<double> output = nn.Compute({6.3,2.3,4.4,1.3});

    std::cout << "Accuracy: " << nn.GetAccuracy(IrisData::dataset) << std::endl;

    std::cin.ignore();
    return 0;
}