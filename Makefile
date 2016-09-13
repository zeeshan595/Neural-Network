

EXE_NAME = Testing
BUILD_DIR = Build/
COMPILER = g++
CFLAGS = -c -Wall
LFLAGS = -Wall

all : $(BUILD_DIR)Testing.so $(BUILD_DIR)ConnectNodes.so $(BUILD_DIR)Activation.so $(BUILD_DIR)LinearNetwork.so $(BUILD_DIR)ParticleSwarmOptimisation.so
		$(COMPILER) $(LFLAGS) $(BUILD_DIR)Testing.so $(BUILD_DIR)ParticleSwarmOptimisation.so $(BUILD_DIR)ConnectNodes.so $(BUILD_DIR)LinearNetwork.so $(BUILD_DIR)Activation.so -o $(BUILD_DIR)$(EXE_NAME)

$(BUILD_DIR)Testing.so : Testing.cpp Structure/LinearNetwork.h Core/Activation.h Core/ConnectNodes.h
		$(COMPILER) $(CFLAGS) Testing.cpp -o $(BUILD_DIR)Testing.so

$(BUILD_DIR)LinearNetwork.so : Structure/LinearNetwork.cpp Structure/LinearNetwork.h Core/Activation.h Core/ConnectNodes.h
		$(COMPILER) $(CFLAGS) Structure/LinearNetwork.cpp -o $(BUILD_DIR)LinearNetwork.so

$(BUILD_DIR)ParticleSwarmOptimisation.so : Learning/ParticleSwarmOptimisation.cpp Learning/ParticleSwarmOptimisation.h
		$(COMPILER) $(CFLAGS) Learning/ParticleSwarmOptimisation.cpp -o $(BUILD_DIR)ParticleSwarmOptimisation.so

$(BUILD_DIR)Activation.so : Core/Activation.cpp Core/Activation.h
		$(COMPILER) $(CFLAGS) Core/Activation.cpp -o $(BUILD_DIR)Activation.so

$(BUILD_DIR)ConnectNodes.so : Core/ConnectNodes.cpp Core/ConnectNodes.h
		$(COMPILER) $(CFLAGS) Core/ConnectNodes.cpp -o $(BUILD_DIR)ConnectNodes.so

clear:
	rm $(BUILD_DIR)*.so $(BUILD_DIR)Testing