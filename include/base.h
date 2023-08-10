#pragma once

#include <glm.hpp>
#include <common.hpp>
#include <gtc/type_ptr.hpp>
#include <gtc/matrix_transform.hpp>
#include <gtc/quaternion.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <gtx/quaternion.hpp>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <regex>
#include <queue>
#include <list>
#include <forward_list>
#include <numeric>
#include <set>
#include <map>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <random>

#include "project_constants.h"

template<typename T>
inline T getRandomValue(T min_value, T max_value)
{
   std::mt19937 random_generator(std::chrono::system_clock::now().time_since_epoch().count());
   std::uniform_real_distribution<T> distribution(min_value, max_value);
   return distribution( random_generator );
}