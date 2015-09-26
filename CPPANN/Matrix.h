#pragma once
#include <vector>

template<typename T>
class Layer
{
public:
	Layer(const std::vector<T> &initialValues) :
		networkValues{ std::move(initialValues) }
	{};

	const std::vector<T> &getValues() const
	{
		return networkValues;
	};

	void applySigmoid() {
		for (auto & networkValue : networkValues) {
			networkValue = sigmoid(networkValue);
		}
	};

private:
	static T sigmoid(T x) {
		return 1 / (1 + exp(-x));
	};

	std::vector<T> networkValues;
};


template<typename T>
class WeightMatrix
{
public:
	WeightMatrix(const std::vector<std::vector<T>> &inputRows) 
	{
		for (auto &inputRow : inputRows) {
			auto weightsBegingIt = weights.end();
			weights.insert(weightsBegingIt, inputRow.begin(), inputRow.end());
		}
	}

	Layer<T> operator*(const Layer<T> &layer)
	{
		std::vector<T> retval;

		//initial conditions
		retval.push_back(0);
		std::vector<T> input = layer.getValues();
		auto inputIt = input.begin();
		

		//loop through the weights and produce a product
		for (auto dataIt = weights.begin(); dataIt != weights.end(); ++dataIt) {
			if (inputIt == input.end())	{
				inputIt = input.begin();
				retval.push_back(0);
			}
			retval.back() += (*inputIt)*(*dataIt);
			++inputIt;
		}
		return Layer<T>{std::move(retval)};
	}

private:
	std::vector<T> weights;
};

