
EXE_NAME 	= Testing
BUILD_DIR 	= Build/
COMPILER 	= g++
CFLAGS 		= -c -Wall -Wc++0x-compat
LFLAGS 		= -Wall
OBJS		= $(BUILD_DIR)Testing.so $(BUILD_DIR)ParticleSwarmOptimisation.so $(BUILD_DIR)GeneticAlgorithm.so $(BUILD_DIR)ConnectNodes.so $(BUILD_DIR)LinearNetwork.so $(BUILD_DIR)RecurrentNetwork.so $(BUILD_DIR)ConvolutionalNetwork.so $(BUILD_DIR)ConvolutionalNetwork2D.so $(BUILD_DIR)Activation.so $(BUILD_DIR)CommonFunctions.so

all : $(OBJS)
	$(COMPILER) $(LFLAGS) $(OBJS) -o $(BUILD_DIR)$(EXE_NAME)

$(BUILD_DIR)Testing.so : Testing.cpp Structure/LinearNetwork.h Core/Activation.h Core/ConnectNodes.h
	$(COMPILER) $(CFLAGS) Testing.cpp -o $(BUILD_DIR)Testing.so

#Structures

$(BUILD_DIR)LinearNetwork.so : Structure/LinearNetwork.cpp Structure/LinearNetwork.h Core/Activation.h Core/ConnectNodes.h
	$(COMPILER) $(CFLAGS) Structure/LinearNetwork.cpp -o $(BUILD_DIR)LinearNetwork.so

$(BUILD_DIR)ConvolutionalNetwork.so : Structure/ConvolutionalNetwork.cpp Structure/ConvolutionalNetwork.h
	$(COMPILER) $(CFLAGS) Structure/ConvolutionalNetwork.cpp -o $(BUILD_DIR)ConvolutionalNetwork.so

$(BUILD_DIR)ConvolutionalNetwork2D.so : Structure/ConvolutionalNetwork2D.cpp Structure/ConvolutionalNetwork2D.h
	$(COMPILER) $(CFLAGS) Structure/ConvolutionalNetwork2D.cpp -o $(BUILD_DIR)ConvolutionalNetwork2D.so

$(BUILD_DIR)RecurrentNetwork.so : Structure/RecurrentNetwork.cpp Structure/RecurrentNetwork.h
	$(COMPILER) $(CFLAGS) Structure/RecurrentNetwork.cpp -o $(BUILD_DIR)RecurrentNetwork.so

#Learning

$(BUILD_DIR)ParticleSwarmOptimisation.so : Learning/ParticleSwarmOptimisation.cpp Learning/ParticleSwarmOptimisation.h
	$(COMPILER) $(CFLAGS) Learning/ParticleSwarmOptimisation.cpp -o $(BUILD_DIR)ParticleSwarmOptimisation.so

$(BUILD_DIR)GeneticAlgorithm.so : Learning/GeneticAlgorithm.cpp Learning/GeneticAlgorithm.h
	$(COMPILER) $(CFLAGS) Learning/GeneticAlgorithm.cpp -o $(BUILD_DIR)GeneticAlgorithm.so

#Core

$(BUILD_DIR)Activation.so : Core/Activation.cpp Core/Activation.h
	$(COMPILER) $(CFLAGS) Core/Activation.cpp -o $(BUILD_DIR)Activation.so

$(BUILD_DIR)ConnectNodes.so : Core/ConnectNodes.cpp Core/ConnectNodes.h
	$(COMPILER) $(CFLAGS) Core/ConnectNodes.cpp -o $(BUILD_DIR)ConnectNodes.so

$(BUILD_DIR)CommonFunctions.so : Core/CommonFunctions.cpp Core/CommonFunctions.h
	$(COMPILER) $(CFLAGS) Core/CommonFunctions.cpp -o $(BUILD_DIR)CommonFunctions.so

#Other

clear:
	rm $(BUILD_DIR)*.so $(BUILD_DIR)Testing