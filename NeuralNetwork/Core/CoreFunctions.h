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

#endif