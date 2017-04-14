#ifndef _GENETIC_ALGORITHM
#define _GENETIC_ALGORITHM

namespace GA
{
    struct Agent
    {
        //The weights and bias of a neural network
        //also known as the dna of the agent
        std::vector<double>     dna;
        //The current error of this agent
        double                  error;
    };

    void Train(
        //The data to train
        std::vector<std::vector<double> >       train_data,
        //The population of agents that should be initialised
        uint32_t                                population,
        //How much should agents be mutated by
        double                                  mutation_amount,
        //The max epoch or generations training should repeat.
        uint32_t                                generations_amount,
        //Pointer to the network that needs to be trained.
        BaseNetwork*                            base_network
    );
};

#include "GA.cpp"
#endif