#include "NeuralNetwork/Common.h"

int main()
{
    std::cout << "Normalizing data..." << std::endl;
    Normalize(IrisData::dataset, {0, 1, 2, 3 });
    Normalize(WineData::dataset, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    Normalize(BreastCancerData::dataset, {0, 1, 2, 3, 4, 5, 6 ,7, 8});
    std::cout << "Creating Neural Network" << std::endl;

    MFNN nn = MFNN({ 21, 4, 4 });

    //nn.SetWeights(IrisData::GOOD_WEIGHTS);
    //PSO::Train(CarData::dataset, 20, 0.05, 1000, dynamic_cast<BaseNetwork*>(&nn));
    //GA::Train(WineData::dataset, 20, 1.0f, 1000, dynamic_cast<BaseNetwork*>(&nn));
    //nn.TrainUsingBP(CarData::dataset, 0.05, 0.01, 0.00001, 2000);
    //std::vector<double> output = nn.Compute({1,14.1,2.16,2.3,18,105,2.95,3.32,.22,2.38,5.75,1.25,3.17});

    //std::cout << "Accuracy: " << nn.GetAccuracy(WineData::dataset) << std::endl;

    PrintDataSet(BreastCancerData::dataset, 5);

    std::cin.ignore();
    return 0;
}