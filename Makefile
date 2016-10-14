
EXE_NAME 	= Testing
BUILD_DIR 	= Build/
COMPILER 	= g++
CFLAGS 		= -c -Wall -Wc++0x-compat -std=c++11
LFLAGS 		= -Wall
OBJS		= $(BUILD_DIR)Testing.so $(BUILD_DIR)Activation.so $(BUILD_DIR)BaseNetwork.so $(BUILD_DIR)LinearNetwork.so $(BUILD_DIR)PSO.so

all : $(OBJS)
	$(COMPILER) $(LFLAGS) $(OBJS) -o $(BUILD_DIR)$(EXE_NAME)

$(BUILD_DIR)Testing.so : Testing.cpp
	$(COMPILER) $(CFLAGS) Testing.cpp -o $(BUILD_DIR)Testing.so

#Core
$(BUILD_DIR)Activation.so : Core/Activation.cpp Core/Activation.h
	$(COMPILER) $(CFLAGS) Core/Activation.cpp -o $(BUILD_DIR)Activation.so

#Structre
$(BUILD_DIR)BaseNetwork.so : Structure/BaseNetwork.h Structure/BaseNetwork.cpp
	$(COMPILER) $(CFLAGS) Structure/BaseNetwork.cpp -o $(BUILD_DIR)BaseNetwork.so

$(BUILD_DIR)LinearNetwork.so : Structure/LinearNetwork.cpp Structure/LinearNetwork.h
	$(COMPILER) $(CFLAGS) Structure/LinearNetwork.cpp -o $(BUILD_DIR)LinearNetwork.so

#Learning
$(BUILD_DIR)PSO.so : Learning/PSO.cpp Learning/PSO.h
	$(COMPILER) $(CFLAGS) Learning/PSO.cpp -o $(BUILD_DIR)PSO.so

#$(BUILD_DIR)BP.so : Learning/BP.cpp Learning/BP.h
#	$(COMPILER) $(CFLAGS) Learning/BP.cpp -o $(BUILD_DIR)BP.so

#Other

clear:
	rm $(BUILD_DIR)*.so $(BUILD_DIR)Testing