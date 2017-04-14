#include "NeuralNetwork/Common.h"

int main()
{
    std::cout << "Normalizing data..." << std::endl;
    Normalize(IrisData::dataset, {0, 1, 2, 3 });
    Normalize(WineData::dataset, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    Normalize(BreastCancerData::dataset, {0, 1, 2, 3, 4, 5, 6 ,7, 8});
    std::cout << "Randomizing order of the data..." << std::endl;
    RandomizeDataSetOrder(IrisData::dataset);
    RandomizeDataSetOrder(BreastCancerData::dataset);
    RandomizeDataSetOrder(WineData::dataset);
    RandomizeDataSetOrder(CarEvaluationData::dataset);

    uint32_t inputs = 21;
    uint32_t hidden = 13;
    uint32_t output = 4;

    MFNN nn = MFNN({ inputs, hidden, output });
    std::cout << "Creating Neural Network ("<<inputs<<", "<<hidden<<", "<<output<<")" << std::endl;

    std::cout << "Current weights set to: " << std::endl;
    std::vector<double> n_weights = nn.GetWeights();
    for (uint32_t i = 0; i < n_weights.size(); i++)
    {
        std::cout << n_weights[i] << ", ";
    }
    std::cout << std::endl << std::endl << std::endl;

    std::cout << "Starting training using GA on Car Evaluation Set" << std::endl;
    std::cout << "Press enter to start the training." << std::endl;
    std::cin.ignore();

    //nn.TrainUsingBP(CarEvaluationData::dataset, 0.05, 0.01, 0.00001, 1000);
    //PSO::Train(CarEvaluationData::dataset, 20, 1000, dynamic_cast<BaseNetwork*>(&nn));
    GA::Train(CarEvaluationData::dataset, 20, 1.0f, 1000, dynamic_cast<BaseNetwork*>(&nn));
    //std::vector<double> output = nn.Compute({1,14.1,2.16,2.3,18,105,2.95,3.32,.22,2.38,5.75,1.25,3.17});

    //std::cout << "Accuracy: " << nn.GetAccuracy(WineData::dataset) << std::endl;

    std::cout << "Done! Press enter to close the program." << std::endl;
    std::cin.ignore();
    return 0;
}