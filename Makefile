
EXE_NAME 	= Testing
BUILD_DIR 	= Build/
COMPILER 	= g++
CFLAGS 		= -c -Wall -Wc++0x-compat -std=c++11
LFLAGS 		= -Wall
OBJS		= $(BUILD_DIR)Testing.so $(BUILD_DIR)LinearNetwork.so $(BUILD_DIR)PSO.so $(BUILD_DIR)Activation.so

all : $(OBJS)
	$(COMPILER) $(LFLAGS) $(OBJS) -o $(BUILD_DIR)$(EXE_NAME)

$(BUILD_DIR)Testing.so : Testing.cpp
	$(COMPILER) $(CFLAGS) Testing.cpp -o $(BUILD_DIR)Testing.so

#Structure

$(BUILD_DIR)LinearNetwork.so : Structure/LinearNetwork.h Structure/LinearNetwork.cpp
	$(COMPILER) $(CFLAGS) Structure/LinearNetwork.cpp -o $(BUILD_DIR)LinearNetwork.so

#Learning

$(BUILD_DIR)PSO.so : Learning/PSO.h Learning/PSO.cpp
	$(COMPILER) $(CFLAGS) Learning/PSO.cpp -o $(BUILD_DIR)PSO.so

#Core

$(BUILD_DIR)Activation.so : Core/Activation.h Core/Activation.cpp
	$(COMPILER) $(CFLAGS) Core/Activation.cpp -o $(BUILD_DIR)Activation.so

#Other

clear:
	rm $(BUILD_DIR)*.so $(BUILD_DIR)Testing