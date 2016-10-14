#ifndef _ACTIVATION_TYPE
#define _ACTIVATION_TYPE

namespace Core
{
    enum ActivationType
    {
        NONE,
        LOGISTIC_SIGMOID,
        HYPERBOLIC_TANGENT,
        HEAVISIDE_STEP,
        SOFTMAX
    };
};

#endif