
#ifdef __MINGW32__
#include <windows.h>
#endif

//GENERAL
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <ctime>
#include <math.h>
#include <chrono>
#include <iomanip> 

//CORE
#include "Core/Activation.h"
#include "Core/CoreFunctions.h"

//STRUCTURE
#include "Structure/Synapse.h"
#include "Structure/Neuron.h"
#include "Structure/BaseNetwork.h"

//NETWORKS
#include "Networks/MFNN.h"

//TRAINNING
#include "Trainning/PSO.h"
#include "Trainning/GA.h"

//DATA SETS
#include "DataSets/Iris.h"
#include "DataSets/Wine.h"
#include "DataSets/Car.h"