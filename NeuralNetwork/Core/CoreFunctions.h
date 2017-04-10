#ifndef _CORE_FUNCTIONS
#define _CORE_FUNCTIONS

std::vector<uint32_t> Shuffle(std::vector<uint32_t> sequence)
{
    std::vector<uint32_t> result = sequence;
	std::srand(std::time(0));
	for (unsigned int i = 0; i < result.size(); ++i)
	{
		int r = std::rand() % result.size();
		int tmp = result[r];
		result[r] = result[i];
		result[i] = tmp;
	}
	return result;
}

void PrintDataSet(std::vector<std::vector<double> > dataset, uint32_t tab)
{
	for (uint32_t i = 0; i < dataset.size(); i++)
    {
        std::cout << "{ ";
        for (uint32_t j = 0; j < dataset.size(); j++)
        {
            std::cout << std::setw(tab) << dataset[i][j] << ", ";
        }
        std::cout << "}," << std::endl;
    }
}

std::vector<std::vector<double> > Normalize(std::vector<std::vector<double> > train_data, uint32_t cols)
{
    for (uint32_t i = 0; i < cols; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < train_data.size(); j++)
            sum += train_data[j][i];

        double mean = sum / train_data.size();
        sum = 0.0;

        for (int j = 0; j < train_data.size(); j++)
            sum += (train_data[j][i] - mean) * (train_data[j][i] - mean);

        double sd = sqrt(sum / (train_data.size() - 1));

        for (int j = 0; j < train_data.size(); j++)
            train_data[j][i] = (train_data[j][i] - mean) / sd;
    }

    return train_data;
}

#endif