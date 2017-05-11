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

    uint32_t inputs = 4;
    uint32_t hidden = 7;
    uint32_t output = 3;

    try
    {
        MFNN nn = MFNN({ inputs, hidden, output });
        std::cout << "Created Neural Network ("<<inputs<<", "<<hidden<<", "<<output<<")" << std::endl;

        std::cout << "Current weights set to: " << std::endl;
        std::vector<double> n_weights = nn.GetWeights();
        for (uint32_t i = 0; i < n_weights.size(); i++)
        {
            std::cout << n_weights[i] << ", ";
        }
        std::cout << std::endl << std::endl << std::endl;
        std::cout << "Current accuracy: " << nn.GetAccuracy(IrisData::dataset) << std::endl;

        std::cout << "Starting training using PSO on Car Evaluation Set" << std::endl;
        std::cout << "Press enter to start the training." << std::endl;
        std::cin.ignore();

        nn.TrainUsingBP(IrisData::dataset, 0.05, 0.01, 0.00001, 1000);
        //PSO::Train(IrisData::dataset, 20, 250, dynamic_cast<BaseNetwork*>(&nn));
        //GA::Train(IrisData::dataset, 20, 1.0f, 1000, dynamic_cast<BaseNetwork*>(&nn));
        //std::vector<double> output = nn.Compute({1,14.1,2.16,2.3,18,105,2.95,3.32,.22,2.38,5.75,1.25,3.17});
        std::cout << "Accuracy After Training: " << nn.GetAccuracy(IrisData::dataset) << std::endl;
        
    }
    catch(std::exception& e)
    {
        std::cout << "ERROR: " << e.what() << std::endl;
    }
    std::cout << "Done! Press enter to close the program." << std::endl;
    std::cin.ignore();
    return 0;
}