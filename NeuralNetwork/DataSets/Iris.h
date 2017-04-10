#ifndef _IRIS_DATASET
#define _IRIS_DATASET

namespace IrisData
{
	const std::vector<double> GOOD_WEIGHTS = {
		-6.24388, 10, -10, -10, 9.68381, 10, 10, -10, -2.06582, 6.13957, -0.479258, -1.82205, -9.98176, 0.199313, -10, -6.62423, 1.68808, 0.123798, -0.676269, -10, 3.41882, 1.20598, -1.39019, 5.10291, -2.26213, -1.32441, 6.61477, 0.345021, 0.986209, 1.54915, -2.60002, -4.5882, -0.3196, -3.45096, -3.38032, 3.34489, -2.72878, -0.803161, 10, -0.55176, 6.39354, 10, 9.97832, -6.58292, -5.71315, -0.984417, -6.47856, -0.814072, -8.84718, -1.28892, -2.20913, 6.44553, 10, 3.26917, -10, 10, -10, 1.17383, 2.17091, -9.99571, -0.102128, -3.03252, -0.856342
	};

	std::vector<std::vector<double> > dataset = {
		{    -0.897674,       1.0156,     -1.33575,     -1.31105,            1,            0,            0, },
		{      -1.1392,    -0.131539,     -1.33575,     -1.31105,            1,            0,            0, },
		{     -1.38073,     0.327318,      -1.3924,     -1.31105,            1,            0,            0, },
		{     -1.50149,    0.0978893,      -1.2791,     -1.31105,            1,            0,            0, },
		{     -1.01844,      1.24503,     -1.33575,     -1.31105,            1,            0,            0, },
		{    -0.535384,      1.93331,     -1.16581,     -1.04867,            1,            0,            0, },
		{     -1.50149,     0.786174,     -1.33575,     -1.17986,            1,            0,            0, },
		{     -1.01844,     0.786174,      -1.2791,     -1.31105,            1,            0,            0, },
		{     -1.74302,    -0.360967,     -1.33575,     -1.31105,            1,            0,            0, },
		{      -1.1392,    0.0978893,      -1.2791,     -1.44224,            1,            0,            0, },
		{    -0.535384,      1.47446,      -1.2791,     -1.31105,            1,            0,            0, },
		{     -1.25996,     0.786174,     -1.22246,     -1.31105,            1,            0,            0, },
		{     -1.25996,    -0.131539,     -1.33575,     -1.44224,            1,            0,            0, },
		{     -1.86378,    -0.131539,     -1.50569,     -1.44224,            1,            0,            0, },
		{   -0.0523308,      2.16274,     -1.44905,     -1.31105,            1,            0,            0, },
		{    -0.173094,      3.08046,      -1.2791,     -1.04867,            1,            0,            0, },
		{    -0.535384,      1.93331,      -1.3924,     -1.04867,            1,            0,            0, },
		{    -0.897674,       1.0156,     -1.33575,     -1.17986,            1,            0,            0, },
		{    -0.173094,      1.70389,     -1.16581,     -1.17986,            1,            0,            0, },
		{    -0.897674,      1.70389,      -1.2791,     -1.17986,            1,            0,            0, },
		{    -0.535384,     0.786174,     -1.16581,     -1.31105,            1,            0,            0, },
		{    -0.897674,      1.47446,      -1.2791,     -1.04867,            1,            0,            0, },
		{     -1.50149,      1.24503,     -1.56234,     -1.31105,            1,            0,            0, },
		{    -0.897674,     0.556746,     -1.16581,    -0.917474,            1,            0,            0, },
		{     -1.25996,     0.786174,     -1.05251,     -1.31105,            1,            0,            0, },
		{     -1.01844,    -0.131539,     -1.22246,     -1.31105,            1,            0,            0, },
		{     -1.01844,     0.786174,     -1.22246,     -1.04867,            1,            0,            0, },
		{    -0.776911,       1.0156,      -1.2791,     -1.31105,            1,            0,            0, },
		{    -0.776911,     0.786174,     -1.33575,     -1.31105,            1,            0,            0, },
		{     -1.38073,     0.327318,     -1.22246,     -1.31105,            1,            0,            0, },
		{     -1.25996,    0.0978893,     -1.22246,     -1.31105,            1,            0,            0, },
		{    -0.535384,     0.786174,      -1.2791,     -1.04867,            1,            0,            0, },
		{    -0.776911,      2.39217,      -1.2791,     -1.44224,            1,            0,            0, },
		{    -0.414621,       2.6216,     -1.33575,     -1.31105,            1,            0,            0, },
		{      -1.1392,    0.0978893,      -1.2791,     -1.31105,            1,            0,            0, },
		{     -1.01844,     0.327318,     -1.44905,     -1.31105,            1,            0,            0, },
		{    -0.414621,       1.0156,      -1.3924,     -1.31105,            1,            0,            0, },
		{      -1.1392,      1.24503,     -1.33575,     -1.44224,            1,            0,            0, },
		{     -1.74302,    -0.131539,      -1.3924,     -1.31105,            1,            0,            0, },
		{    -0.897674,     0.786174,      -1.2791,     -1.31105,            1,            0,            0, },
		{     -1.01844,       1.0156,      -1.3924,     -1.17986,            1,            0,            0, },
		{     -1.62225,     -1.73754,      -1.3924,     -1.17986,            1,            0,            0, },
		{     -1.74302,     0.327318,      -1.3924,     -1.31105,            1,            0,            0, },
		{     -1.01844,       1.0156,     -1.22246,    -0.786281,            1,            0,            0, },
		{    -0.897674,      1.70389,     -1.05251,     -1.04867,            1,            0,            0, },
		{     -1.25996,    -0.131539,     -1.33575,     -1.17986,            1,            0,            0, },
		{    -0.897674,      1.70389,     -1.22246,     -1.31105,            1,            0,            0, },
		{     -1.50149,     0.327318,     -1.33575,     -1.31105,            1,            0,            0, },
		{    -0.656147,      1.47446,      -1.2791,     -1.31105,            1,            0,            0, },
		{     -1.01844,     0.556746,     -1.33575,     -1.31105,            1,            0,            0, },
		{      1.39683,     0.327318,     0.533621,      0.26326,            0,            1,            0, },
		{     0.672249,     0.327318,     0.420326,     0.394453,            0,            1,            0, },
		{      1.27607,    0.0978893,     0.646916,     0.394453,            0,            1,            0, },
		{    -0.414621,     -1.73754,     0.137087,     0.132067,            0,            1,            0, },
		{     0.793012,    -0.590395,     0.476973,     0.394453,            0,            1,            0, },
		{    -0.173094,    -0.590395,     0.420326,     0.132067,            0,            1,            0, },
		{     0.551486,     0.556746,     0.533621,     0.525645,            0,            1,            0, },
		{      -1.1392,     -1.50811,    -0.259446,    -0.261511,            0,            1,            0, },
		{     0.913776,    -0.360967,     0.476973,     0.132067,            0,            1,            0, },
		{    -0.776911,    -0.819823,    0.0804397,      0.26326,            0,            1,            0, },
		{     -1.01844,     -2.42582,    -0.146151,    -0.261511,            0,            1,            0, },
		{    0.0684325,    -0.131539,     0.250383,     0.394453,            0,            1,            0, },
		{     0.189196,     -1.96696,     0.137087,    -0.261511,            0,            1,            0, },
		{     0.309959,    -0.360967,     0.533621,      0.26326,            0,            1,            0, },
		{    -0.293857,    -0.360967,   -0.0895033,     0.132067,            0,            1,            0, },
		{      1.03454,    0.0978893,     0.363678,      0.26326,            0,            1,            0, },
		{    -0.293857,    -0.131539,     0.420326,     0.394453,            0,            1,            0, },
		{   -0.0523308,    -0.819823,     0.193735,    -0.261511,            0,            1,            0, },
		{     0.430722,     -1.96696,     0.420326,     0.394453,            0,            1,            0, },
		{    -0.293857,     -1.27868,    0.0804397,    -0.130318,            0,            1,            0, },
		{    0.0684325,     0.327318,     0.590269,     0.788031,            0,            1,            0, },
		{     0.309959,    -0.590395,     0.137087,     0.132067,            0,            1,            0, },
		{     0.551486,     -1.27868,     0.646916,     0.394453,            0,            1,            0, },
		{     0.309959,    -0.590395,     0.533621,  0.000874618,            0,            1,            0, },
		{     0.672249,    -0.360967,      0.30703,     0.132067,            0,            1,            0, },
		{     0.913776,    -0.131539,     0.363678,      0.26326,            0,            1,            0, },
		{       1.1553,    -0.590395,     0.590269,      0.26326,            0,            1,            0, },
		{      1.03454,    -0.131539,     0.703564,     0.656838,            0,            1,            0, },
		{     0.189196,    -0.360967,     0.420326,     0.394453,            0,            1,            0, },
		{    -0.173094,     -1.04925,    -0.146151,    -0.261511,            0,            1,            0, },
		{    -0.414621,     -1.50811,     0.023792,    -0.130318,            0,            1,            0, },
		{    -0.414621,     -1.50811,   -0.0328556,    -0.261511,            0,            1,            0, },
		{   -0.0523308,    -0.819823,    0.0804397,  0.000874618,            0,            1,            0, },
		{     0.189196,    -0.819823,     0.760211,     0.525645,            0,            1,            0, },
		{    -0.535384,    -0.131539,     0.420326,     0.394453,            0,            1,            0, },
		{     0.189196,     0.786174,     0.420326,     0.525645,            0,            1,            0, },
		{      1.03454,    0.0978893,     0.533621,     0.394453,            0,            1,            0, },
		{     0.551486,     -1.73754,     0.363678,     0.132067,            0,            1,            0, },
		{    -0.293857,    -0.131539,     0.193735,     0.132067,            0,            1,            0, },
		{    -0.414621,     -1.27868,     0.137087,     0.132067,            0,            1,            0, },
		{    -0.414621,     -1.04925,     0.363678,  0.000874618,            0,            1,            0, },
		{     0.309959,    -0.131539,     0.476973,      0.26326,            0,            1,            0, },
		{   -0.0523308,     -1.04925,     0.137087,  0.000874618,            0,            1,            0, },
		{     -1.01844,     -1.73754,    -0.259446,    -0.261511,            0,            1,            0, },
		{    -0.293857,    -0.819823,     0.250383,     0.132067,            0,            1,            0, },
		{    -0.173094,    -0.131539,     0.250383,  0.000874618,            0,            1,            0, },
		{    -0.173094,    -0.360967,     0.250383,     0.132067,            0,            1,            0, },
		{     0.430722,    -0.360967,      0.30703,     0.132067,            0,            1,            0, },
		{    -0.897674,     -1.27868,    -0.429389,    -0.130318,            0,            1,            0, },
		{    -0.173094,    -0.590395,     0.193735,     0.132067,            0,            1,            0, },
		{     0.551486,     0.556746,      1.27004,      1.70638,            0,            0,            1, },
		{   -0.0523308,    -0.819823,     0.760211,     0.919223,            0,            0,            1, },
		{      1.51759,    -0.131539,      1.21339,      1.18161,            0,            0,            1, },
		{     0.551486,    -0.360967,      1.04345,     0.788031,            0,            0,            1, },
		{     0.793012,    -0.131539,      1.15675,       1.3128,            0,            0,            1, },
		{      2.12141,    -0.131539,      1.60993,      1.18161,            0,            0,            1, },
		{      -1.1392,     -1.27868,     0.420326,     0.656838,            0,            0,            1, },
		{      1.75912,    -0.360967,      1.43998,     0.788031,            0,            0,            1, },
		{      1.03454,     -1.27868,      1.15675,     0.788031,            0,            0,            1, },
		{      1.63836,      1.24503,      1.32669,      1.70638,            0,            0,            1, },
		{     0.793012,     0.327318,     0.760211,      1.05042,            0,            0,            1, },
		{     0.672249,    -0.819823,     0.873507,     0.919223,            0,            0,            1, },
		{       1.1553,    -0.131539,     0.986802,      1.18161,            0,            0,            1, },
		{    -0.173094,     -1.27868,     0.703564,      1.05042,            0,            0,            1, },
		{   -0.0523308,    -0.590395,     0.760211,      1.57519,            0,            0,            1, },
		{     0.672249,     0.327318,     0.873507,      1.44399,            0,            0,            1, },
		{     0.793012,    -0.131539,     0.986802,     0.788031,            0,            0,            1, },
		{      2.24217,      1.70389,      1.66657,       1.3128,            0,            0,            1, },
		{      2.24217,     -1.04925,      1.77987,      1.44399,            0,            0,            1, },
		{     0.189196,     -1.96696,     0.703564,     0.394453,            0,            0,            1, },
		{      1.27607,     0.327318,       1.1001,      1.44399,            0,            0,            1, },
		{    -0.293857,    -0.590395,     0.646916,      1.05042,            0,            0,            1, },
		{      2.24217,    -0.590395,      1.66657,      1.05042,            0,            0,            1, },
		{     0.551486,    -0.819823,     0.646916,     0.788031,            0,            0,            1, },
		{      1.03454,     0.556746,       1.1001,      1.18161,            0,            0,            1, },
		{      1.63836,     0.327318,      1.27004,     0.788031,            0,            0,            1, },
		{     0.430722,    -0.590395,     0.590269,     0.788031,            0,            0,            1, },
		{     0.309959,    -0.131539,     0.646916,     0.788031,            0,            0,            1, },
		{     0.672249,    -0.590395,      1.04345,      1.18161,            0,            0,            1, },
		{      1.63836,    -0.131539,      1.15675,     0.525645,            0,            0,            1, },
		{      1.87988,    -0.590395,      1.32669,     0.919223,            0,            0,            1, },
		{       2.4837,      1.70389,      1.49663,      1.05042,            0,            0,            1, },
		{     0.672249,    -0.590395,      1.04345,       1.3128,            0,            0,            1, },
		{     0.551486,    -0.590395,     0.760211,     0.394453,            0,            0,            1, },
		{     0.309959,     -1.04925,      1.04345,      0.26326,            0,            0,            1, },
		{      2.24217,    -0.131539,      1.32669,      1.44399,            0,            0,            1, },
		{     0.551486,     0.786174,      1.04345,      1.57519,            0,            0,            1, },
		{     0.672249,    0.0978893,     0.986802,     0.788031,            0,            0,            1, },
		{     0.189196,    -0.131539,     0.590269,     0.788031,            0,            0,            1, },
		{      1.27607,    0.0978893,     0.930154,      1.18161,            0,            0,            1, },
		{      1.03454,    0.0978893,      1.04345,      1.57519,            0,            0,            1, },
		{      1.27607,    0.0978893,     0.760211,      1.44399,            0,            0,            1, },
		{   -0.0523308,    -0.819823,     0.760211,     0.919223,            0,            0,            1, },
		{       1.1553,     0.327318,      1.21339,      1.44399,            0,            0,            1, },
		{      1.03454,     0.556746,       1.1001,      1.70638,            0,            0,            1, },
		{      1.03454,    -0.131539,     0.816859,      1.44399,            0,            0,            1, },
		{     0.551486,     -1.27868,     0.703564,     0.919223,            0,            0,            1, },
		{     0.793012,    -0.131539,     0.816859,      1.05042,            0,            0,            1, },
		{     0.430722,     0.786174,     0.930154,      1.44399,            0,            0,            1, },
		{    0.0684325,    -0.131539,     0.760211,     0.788031,            0,            0,            1, },
	};
};
#endif