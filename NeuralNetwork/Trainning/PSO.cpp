void PSO::Train(
        //training data 
        std::vector<std::vector<double> >   train_data,
        //amount of particles to use for training
        uint32_t                            particles_count,
        //max epochs this training should be repeated for
        uint32_t                            repeat,
        //A pointer to the neural network that
        //needs to be trained.
        BaseNetwork*                        base_network
)
{
    //Error Checking
    //There cannot be less than 1 particle
    if (particles_count < 1)
        throw std::runtime_error("ERROR [PSO::Train]: Particle count must be greater than 0");
    //max epochs cannot be smaller then 1
    if (repeat < 1)
        throw std::runtime_error("ERROR [PSO::Train]: Repeat must be greater than 0");
    //The pointer to the neural network cannot be equal to NULL
    if (base_network == NULL || base_network == nullptr)
        throw std::runtime_error("ERROR [PSO::Train]: BaseNetwork cannot be NULL");
    //Make sure the training data does not equal NULL
    if (train_data.size() <= 0)
        throw std::runtime_error("ERROR [PSO::Train]: Train data is does not contain any data");


    //Setup min and max position for particles
    double                  MIN                     =-10.0;
    double                  MAX                     =+10.0;
    //Setup free parameters
    double                  inertia_weight          = 0.729;
    double                  cognitive_weight        = 1.49445;
    double                  social_weight           = 1.49445;
    //Store current weights and current weight length in a variable
    std::vector<double>     current_weights         = base_network->GetWeights();
    uint32_t                weights_length          = current_weights.size();
    //setup epoch counter
    uint32_t                repeat_counter          = 0;
    //Setup variables to be used as random number generators
    double                  r1                      = 0; //Random Number 1
    double                  r2                      = 0; //Random Number 2
    //Setup global variables (best global position, best global error)
    double                  best_global_error       = std::numeric_limits<double>::max();
    std::vector<double>     best_global_position(weights_length);
    //create a sequence variable for randomizing which particle is processed first.
    std::vector<uint32_t>   sequence(particles_count);
    //create a swarm of particles.
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
        //Change the size of position and velocity to match weights
        std::vector<double> position(weights_length);
        std::vector<double> velocity(weights_length);
        //Use min and max to create a low and high number for the
        //number generator
        double low  = 0.1 * MIN;
        double high = 0.1 * MAX;
        for (uint32_t j = 0; j < weights_length; j++)
        {
            //Use random numbers to compute a velocity and position
            velocity[j] = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
            position[j] = (high - low) * ((double)std::rand() / (double)RAND_MAX) + low;
        }
        //Check the error of the current particle
        double error    = base_network->GetMeanSquaredError(train_data, position);
        //Initialise the particle & set all the variables
        swarm[i]        = Particle();
        swarm[i].position          = position;
        swarm[i].velocity          = velocity;
        swarm[i].best_position     = position;
        swarm[i].error             = error;
        swarm[i].best_error        = error;

        //If the particles best error is smaller then best global error
        if (swarm[i].best_error < best_global_error)
        {
            //Then replace the best global position and error with
            //the particles bbest position and error
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
        auto begin = std::chrono::high_resolution_clock::now();
        
        //Rearange the particles in a random order
        sequence = Shuffle(sequence);

        //Create new varibles where the newly computed particle's
        //position, velocity and error will be stored
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
                //using MIN and MAX variables
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
        }

        //get current time and subtract it from the time noted in "begin" variable
        auto elapsed_secs = std::chrono::high_resolution_clock::now() - begin;
        //Convert time to microseconds
        long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed_secs).count();
        //Output text to the console.
        base_network->SetWeights(best_global_position);
        std::cout << "epoch: " << repeat_counter << ", accuracy:" << base_network->GetAccuracy(train_data) << std::endl;

        repeat_counter++;
    }

    base_network->SetWeights(best_global_position);
}