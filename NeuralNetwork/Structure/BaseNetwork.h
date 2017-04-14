#ifndef _BASE_NETWORK
#define _BASE_NETWORK

class BaseNetwork
{
public:
    virtual std::vector<double> Compute(std::vector<double> inputs)
    {
        return {};
    }

    virtual double GetMeanSquaredError(std::vector<std::vector<double> > data, std::vector<double> weights)
    {
        return 100.0;
    }

    virtual double GetAccuracy(std::vector<std::vector<double> > data)
    {
        
    }

    virtual void SetWeights(std::vector<double> weights)
    {

    }

    virtual std::vector<double> GetWeights()
    {
        return {};
    }
};

#endif