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
        for (uint32_t j = 0; j < dataset[i].size(); j++)
        {
            std::cout << std::setw(tab) << dataset[i][j] << ", ";
        }
        std::cout << "}," << std::endl;
    }
}

void Normalize(std::vector<std::vector<double> > &train_data, std::vector<uint32_t> cols)
{
    for (uint32_t col = 0; col < cols.size(); col++)
    {
        uint32_t i = cols[col];
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
}

void RandomizeDataSetOrder(std::vector<std::vector<double> > &train_data)
{
    //Use shuffle function to create a randomly ordered index for
    //the training data.
    std::vector<uint32_t> sequence(train_data.size());
    for (uint32_t i = 0; i < train_data.size(); i++)
        sequence[i] = i;
    sequence = Shuffle(sequence);

    //Use the randomly created index to store randomly ordered train data
    //into new_train_data variable
    std::vector<std::vector<double> > new_train_data(train_data.size());
    for (uint32_t i = 0; i < train_data.size(); i++)
    {
        new_train_data[i] = train_data[sequence[i]];
    }

    //Update the training data.
    train_data = new_train_data;
}

#endif