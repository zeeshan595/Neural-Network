#ifndef _GENETIC_ALGORITHM
#define _GENETIC_ALGORITHM

namespace GA
{
    struct Agent
    {
        std::vector<double>     dna;
        double                  error;
    };

    void Train(
        std::vector<std::vector<double> >       train_data,
        uint32_t                                population,
        double                                  mutation_amount,
        uint32_t                                generations_amount,
        BaseNetwork*                            base_network
    );
};

#include "GA.cpp"
#endif