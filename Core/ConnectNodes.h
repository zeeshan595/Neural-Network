#ifndef _CONNECT_NODES
#define _CONNECT_NODES

#include <vector>

namespace Core
{
	enum ConnectionType
	{
		CONNECT_ALL,
		RANDOM_CONNECTIONS,
		HALF_AND_HALF,
		NO_CONNECTIONS
	};

	class ConnectionMethod
	{
	public:
		static std::vector<std::vector<bool> > ConnectNodes(std::vector<std::vector<bool> > nodeConnections, ConnectionType type);
	};
};

#endif