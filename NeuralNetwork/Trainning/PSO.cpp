void PSO::Train(
        std::vector<std::vector<double> >   train_data,
        uint32_t                            particles_count,
        double                              exit_error,
        double                              death_probability,
        uint32_t                            repeat,
        BaseNetwork*                        base_network
)
{
    //Error Checking
    if (particles_count < 1)
        throw std::runtime_error("ERROR [PSO::Train]: Particle count must be greater than 0");
    if (repeat < 1)
        throw std::runtime_error("ERROR [PSO::Train]: Repeat must be greater than 0");
    if (base_network == NULL || base_network == nullptr)
        throw std::runtime_error("ERROR [PSO::Train]: BaseNetwork cannot be NULL");

    //Setup
    double                  MIN                     =-10.0;
    double                  MAX                     =+10.0;
    double                  inertia_weight          = 0.729;
    double                  cognitive_weight        = 1.49445;
    double                  social_weight           = 1.49445;
    std::vector<double>     current_weights         = base_network->GetWeights();
    uint32_t                weights_length          = current_weights.size();
    uint32_t                repeat_counter          = 0;
    double                  r1                      = 0; //Random Number 1
    double                  r2                      = 0; //Random Number 2
    double                  best_global_error       = std::numeric_limits<double>::max();
    std::vector<double>     best_global_position(weights_length);
    std::vector<uint32_t>   sequence(particles_count);
    std::vector<Particle>   swarm(particles_count);

    //Set 1st particle's position to current weights
    //This ensures we can retrain the already trainned neural network
    {
        std::vector<double> velocity(weights_length);
        double low  = 0.1 * MIN;
        double high = 0.1 * MAX;
        for (uint32_t j = 0; j < weights_length; j++)
        {
            velocity[j] = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
        }
        double error    = base_network->GetMeanSquaredError(train_data, current_weights);
        swarm[0]        = Particle();
        swarm[0].position          = current_weights;
        swarm[0].velocity          = velocity;
        swarm[0].best_position     = current_weights;
        swarm[0].error             = error;
        swarm[0].best_error        = error;

        best_global_position        = current_weights;
        best_global_error           = error;
    }

    //Setup all particles
    for (uint32_t i = 1; i < particles_count; i++)
    {
        std::vector<double> position(weights_length);
        std::vector<double> velocity(weights_length);
        double low  = 0.1 * MIN;
        double high = 0.1 * MAX;
        for (uint32_t j = 0; j < weights_length; j++)
        {
            velocity[j] = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
            position[j] = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
        }
        double error    = base_network->GetMeanSquaredError(train_data, position);
        swarm[i]        = Particle();
        swarm[i].position          = position;
        swarm[i].velocity          = velocity;
        swarm[i].best_position     = position;
        swarm[i].error             = error;
        swarm[i].best_error        = error;

        if (swarm[i].best_error < best_global_error)
        {
            best_global_position    = position;
            best_global_error       = swarm[i].best_error;
        }
    }

    //Setup sequence so the particles can be re aranged each epoch
    for (uint32_t i = 0; i < particles_count; i++)
        sequence[i] = i;
    
    //Start trainning
    while (repeat_counter < repeat)
    {
        //Used to measure the duration per epoch
        clock_t begin = clock();

        //Exit if error is too low
        if (best_global_error < exit_error)
            break;
        
        //Rearange the particles in a random order
        sequence = Shuffle(sequence);

        //Setup
        std::vector<double>     new_position(weights_length);
        std::vector<double>     new_velocity(weights_length);
        double                  new_error;

        //go through all the particles
        for (uint32_t p = 0; p < particles_count; p++)
        {
            //Select a particle in the list using random ordering
            uint32_t i = sequence[p];

            //Compute particle velocity
            for (uint32_t j = 0; j < weights_length; j++)
            {
                r1 = (double)std::rand() / (double)RAND_MAX;
				r2 = (double)std::rand() / (double)RAND_MAX;
                new_velocity[j] =   ((inertia_weight  	* swarm[i].velocity[j]) +
							        (cognitive_weight 	* r1 * (swarm[i].best_position[j] 	- swarm[i].position[j])) +
								    (social_weight 		* r2 * (best_global_position[j] 	- swarm[i].position[j])));
            }
            swarm[i].velocity = new_velocity;

            //Compute particle position
            for (uint32_t j = 0; j < weights_length; j++)
            {
                new_position[j] = swarm[i].position[j] + new_velocity[j];
                //Make sure particle does not go out of bounds.
                if (new_position[j] < MIN)
                    new_position[j] = MIN;
                else if (new_position[j] > MAX)
                    new_position[j] = MAX;
            }
            swarm[i].position = new_position;

            //Compute Particle error
            new_error = base_network->GetMeanSquaredError(train_data, new_position);
            swarm[i].error = new_error;

            //Compare current error with best particle error
            if (new_error < swarm[i].best_error)
            {
                swarm[i].best_error     = new_error;
                swarm[i].best_position  = new_position;
            }

            //Compare current error with best global error
            if (new_error < best_global_error)
            {
                best_global_error       = new_error;
                best_global_position    = new_position;
            }

            //Kill Particles Randomly
            /*TODO*/
        }

        //Used to measure the duration per epoch
        clock_t end             = clock();
        double  elapsed_secs    = double(end - begin) / CLOCKS_PER_SEC;
        std::cout << "Trainning: " << repeat_counter << "/" << repeat << ": " << best_global_error << " - " << elapsed_secs << std::endl;
        repeat_counter++;
    }

    base_network->SetWeights(best_global_position);
}