void GA::Train(
    std::vector<std::vector<double> >       train_data,
    uint32_t                                population,
    double                                  mutation_amount,
    uint32_t                                generations_amount,
    BaseNetwork*                            base_network
)
{
    //Error Checking
    //Ensure there are more then or equal to 2 agents
    if (population < 2)
        throw std::runtime_error("ERROR [GA::Train]: Population size too low, must be greater then or equal to 2 for mating");
    //Make sure the population size is a multiple of 2
    if (population % 2 != 0)
        throw std::runtime_error("ERROR [GA::Train]: Population size must be a multiple of 2");
    //Make sure mutation is above zero
    if (mutation_amount < 0)
        throw std::runtime_error("ERROR [GA::Train]: Mutation amount must be a be equal or above 0");
    //Make sure generations amount is bigger than 0
    if (generations_amount < 1)
        throw std::runtime_error("ERROR [GA::Train]: Generations amount must be a be above 0");
    //Make sure the neural network pointer is not NULL
    if (base_network == NULL || base_network == nullptr)
        throw std::runtime_error("ERROR [GA::Train]: Base Network cannot equal NULL");
    //Make sure the training data does not equal NULL
    if (train_data.size() <= 0)
        throw std::runtime_error("ERROR [GA::Train]: Train data is does not contain any data");
    


    //Store currenet weights & bias of the neural network and
    //the length of those weights/bias for easy access
    std::vector<double>     current_weights         = base_network->GetWeights();
    uint32_t                weights_length          = current_weights.size();
    //Setup a epoch counter that keeps track of current epoch
    uint32_t                repeat_counter          = 0;
    //Stores the best agent with the minimum error
    uint32_t                best_agent              = 0;
    //2 variables that are used to compute random numbers
    //so the range of the random number generator is not too large.
    double                  low                     = -0.1;
    double                  high                    = +0.1;
    //This is used to change the order in which agents are processed.
    std::vector<uint32_t>   sequence(population);
    //A list of agents that will be used to train the neural network
    std::vector<Agent>      agents(population);
    //Some extra variables to help with the mating processes
    std::vector<double>     w1(weights_length);     // Used for mating
    std::vector<double>     w2(weights_length);     // Used for mating

    //Compute first agent using current weights
    //This is done so we can re train by having current weights
    //having some impact on the training process.
    {
        agents[0] = Agent();
        agents[0].dna           = current_weights;
        agents[0].error         = base_network->GetMeanSquaredError(train_data, agents[0].dna);
    }
    //For each agent in the list
    for (uint32_t i = 1; i < population; i++)
    {
        agents[i] = Agent();
        //Change the DNA size to match weights and bias
        agents[i].dna.resize(weights_length);
        for (uint32_t j = 0; j < weights_length; j++)
        {
            //Set each weight and bias to a random value
            agents[i].dna[j]    = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
        }
        //Compute the error using the DNA just created by the agent.
        agents[i].error         = base_network->GetMeanSquaredError(train_data, agents[i].dna);
    }

    //Setup sequence so the particles can be re aranged each epoch
    for (uint32_t i = 0; i < population; i++)
        sequence[i] = i;

    while (repeat_counter < generations_amount)
    {
        //Used to measure the duration per epoch
        auto begin = std::chrono::high_resolution_clock::now();


        //Re arrange arrays with the best being first
        //Each agent checkes every other agent from left to right
        //and if that agent is weaker then the agent it is checking
        //they swap places.
        for (uint32_t i = 0; i < population; i++)
        {
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
            //Replace the agents with the first 2 agents in the pack
			agents[sequence[i]].dna = agents[sequence[i % (population / 2)]].dna;
		}

        //Mate Everyone
		for (uint32_t i = 2; i < population / 2; i++)
		{
            //Go through each weight and bias value
			for (uint32_t j = 0; j < weights_length; j++)
			{
                //Compute a random number
				if (std::rand() % 2 == 0)
				{
                    //If the computed random number is even then
                    //make the first child's weight be the same as first parent's
                    //weight and second childs weight be the same as second parent's
                    //weight
					w1[j] = agents[sequence[(i * 2) + 0]].dna[j];
					w2[j] = agents[sequence[(i * 2) + 1]].dna[j];
				}
				else
				{
                    //If the computed random number is odd then
                    //make the first child's weight be the same as second parent's
                    //weight and second childs weight be the same as first parent's
                    //weight
					w1[j] = agents[sequence[(i * 2) + 1]].dna[j];
					w2[j] = agents[sequence[(i * 2) + 0]].dna[j];
				}
			}
            //Replace the parents with the childs created using their DNA
			agents[sequence[(i * 2) + 0]].dna = w1;
			agents[sequence[(i * 2) + 1]].dna = w2;
		}

        //Mutate Randomly
		for (uint32_t i = 2; i < population; i++)
		{
			for (uint32_t j = 0; j < weights_length; j++)
			{
                //Compute a random number between high and low values
                double m = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
                //Apply mutation
				agents[sequence[i]].dna[j] = agents[sequence[i]].dna[j] + (m * mutation_amount);
			}
		}

        //Compute errors
        for (uint32_t i = 0; i < population; i++)
		{
			agents[i].error = base_network->GetMeanSquaredError(train_data, agents[i].dna);
            //If current agent has a better error then best agent
            //then replace it with the best agent.
            if (agents[best_agent].error > agents[i].error)
            {
                best_agent = i;
            }
		}

        //get current time and subtract it from the time noted in "begin" variable
        auto elapsed_secs = std::chrono::high_resolution_clock::now() - begin;
        //Convert time to microseconds
        long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_secs).count();
        //Output text to the console.
        base_network->SetWeights(agents[best_agent].dna);
        std::cout << repeat_counter << "," << agents[best_agent].error << "," << base_network->GetAccuracy(train_data) << "," << microseconds << std::endl;
        repeat_counter++;
    }
    base_network->SetWeights(agents[best_agent].dna);
}