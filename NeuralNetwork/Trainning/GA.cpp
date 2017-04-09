void GA::Train(
    std::vector<std::vector<double> >       train_data,
    uint32_t                                population,
    uint32_t                                mutation_amount,
    uint32_t                                generations_amount,
    BaseNetwork*                            base_network
)
{
    //Error Checking
    if (population < 2)
        throw std::runtime_error("ERROR [GA::Train]: Population size too low, must be greater then or equal to 2 for mating");
    if (mutation_amount < 0)
        throw std::runtime_error("ERROR [GA::Train]: Mutation amount must be a be equal or above 0");
    if (generations_amount < 1)
        throw std::runtime_error("ERROR [GA::Train]: Generations amount must be a be above 0");
    if (base_network == NULL || base_network == nullptr)
        throw std::runtime_error("ERROR [GA::Train]: Base Network cannot equal NULL");

    //Setup
    double                  MIN                     =-10.0;
    double                  MAX                     =+10.0;
    std::vector<double>     current_weights         = base_network->GetWeights();
    uint32_t                weights_length          = current_weights.size();
    uint32_t                repeat_counter          = 0;
    uint32_t                best_agent              = 0;
    double                  low                     = 0.1 * MIN;
    double                  high                    = 0.1 * MAX;
    std::vector<uint32_t>   sequence(population);
    std::vector<Agent>      agents(population);
    std::vector<double>     w1(weights_length);     // Used for mating
    std::vector<double>     w2(weights_length);     // Used for mating

    //Compute first agent using current weights
    {
        agents[0] = Agent();
        agents[0].dna           = current_weights;
        agents[0].error         = base_network->GetMeanSquaredError(train_data, agents[0].dna);
    }
    //Compute rest of the agents
    for (uint32_t i = 1; i < population; i++)
    {
        agents[i] = Agent();
        agents[i].dna.resize(weights_length);
        for (uint32_t j = 0; j < weights_length; j++)
        {
            agents[i].dna[j]    = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
        }
        agents[i].error         = base_network->GetMeanSquaredError(train_data, agents[i].dna);
    }

    //Setup sequence so the particles can be re aranged each epoch
    for (uint32_t i = 0; i < population; i++)
        sequence[i] = i;

    while (repeat_counter < generations_amount)
    {
        //Used to measure the duration per epoch
        clock_t begin = clock();

        //Re arrange arrays with the best being first
        for (uint32_t i = 0; i < population; i++)
        {
            //Re-arrange Arrays with best being first
            for (uint32_t i = 0; i < population; i++)
            {
                for (uint32_t j = i+1; j < population; j++)
                {
                    if (agents[sequence[i]].error > agents[sequence[j]].error)
                    {
                        uint32_t temp = sequence[i];
                        sequence[i] = sequence[j];
                        sequence[j] = temp;
                    }
                }
            }
        }

        //Kill half of the weak population
		for (uint32_t i = (population / 2); i < population; i++)
		{
			agents[sequence[i]].dna = agents[sequence[i % (population / 2)]].dna;
		}

        //Mate Everyone
		for (uint32_t i = 2; i < population / 2; i++)
		{
			for (uint32_t j = 0; j < weights_length; j++)
			{
				if (std::rand() % 2 == 0)
				{
					w1[j] = agents[sequence[(i * 2) + 0]].dna[j];
					w2[j] = agents[sequence[(i * 2) + 1]].dna[j];
				}
				else
				{
					w1[j] = agents[sequence[(i * 2) + 1]].dna[j];
					w2[j] = agents[sequence[(i * 2) + 0]].dna[j];
				}
			}
			agents[sequence[(i * 2) + 0]].dna = w1;
			agents[sequence[(i * 2) + 1]].dna = w2;
		}

        //Mutate Randomly mutation_amount
		for (uint32_t i = 2; i < population; i++)
		{
			for (uint32_t j = 0; j < weights_length; j++)
			{
                double m = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
				agents[sequence[i]].dna[j] = agents[sequence[i]].dna[j] + m;
			}
		}

        //Compute errors
        for (uint32_t i = 0; i < population; i++)
		{
			agents[i].error = base_network->GetMeanSquaredError(train_data, agents[i].dna);
            if (agents[best_agent].error > agents[i].error)
            {
                best_agent = i;
            }
		}

        //Used to measure the duration per epoch
        clock_t end             = clock();
        double  elapsed_secs    = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Trainning " << repeat_counter << "/" << generations_amount << " : " << agents[best_agent].error << " - " << elapsed_secs << std::endl;
        repeat_counter++;
    }
    base_network->SetWeights(agents[best_agent].dna);
}