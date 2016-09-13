#include "ConnectNodes.h"
#include <iostream>

using namespace Core;

std::vector<std::vector<bool> > ConnectionMethod::ConnectNodes(std::vector<std::vector<bool> > nodeConnections, ConnectionType type)
{
	std::vector<std::vector<bool> > nodes;
	nodes.resize(nodeConnections.size());
	for (unsigned int i = 0; i < nodes.size(); i++)
	{
		nodes[i].resize(nodeConnections[i].size());
		for (unsigned int j = 0; j < nodes[i].size(); j++)
		{
			if (type == CONNECT_ALL)
				nodes[i][j] = true;
			else if (type == HALF_AND_HALF)
				nodes[i][j] = (i * nodes[i].size()) + j % 2 == 0;
			else if (type == NO_CONNECTIONS)
				nodes[i][j] = false;
		}
	}

	return nodes;
}