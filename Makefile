
EXE_NAME 	= Testing
BUILD_DIR 	= Build/
COMPILER 	= g++
CFLAGS 		= -c -Wall
LFLAGS 		= -Wall
OBJS		= $(BUILD_DIR)Testing.so $(BUILD_DIR)ParticleSwarmOptimisation.so $(BUILD_DIR)ConnectNodes.so $(BUILD_DIR)LinearNetwork.so $(BUILD_DIR)ConvolutionalNetwork.so $(BUILD_DIR)Activation.so $(BUILD_DIR)CommonFunctions.so

all : $(OBJS)
		$(COMPILER) $(LFLAGS) $(OBJS) -o $(BUILD_DIR)$(EXE_NAME)

$(BUILD_DIR)Testing.so : Testing.cpp Structure/LinearNetwork.h Core/Activation.h Core/ConnectNodes.h
		$(COMPILER) $(CFLAGS) Testing.cpp -o $(BUILD_DIR)Testing.so

#Structures

$(BUILD_DIR)LinearNetwork.so : Structure/LinearNetwork.cpp Structure/LinearNetwork.h Core/Activation.h Core/ConnectNodes.h
		$(COMPILER) $(CFLAGS) Structure/LinearNetwork.cpp -o $(BUILD_DIR)LinearNetwork.so

$(BUILD_DIR)ConvolutionalNetwork.so : Structure/ConvolutionalNetwork.cpp Structure/ConvolutionalNetwork.h
		$(COMPILER) $(CFLAGS) Structure/ConvolutionalNetwork.cpp -o $(BUILD_DIR)ConvolutionalNetwork.so

#Learning

$(BUILD_DIR)ParticleSwarmOptimisation.so : Learning/ParticleSwarmOptimisation.cpp Learning/ParticleSwarmOptimisation.h
		$(COMPILER) $(CFLAGS) Learning/ParticleSwarmOptimisation.cpp -o $(BUILD_DIR)ParticleSwarmOptimisation.so

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