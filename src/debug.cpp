#ifndef DEBUG_H
#define DEBUG_H

#include "debug.h"

#include <set>
#include <string>

std::set<std::string> debug_mode_set_;

bool isDebugMode(std::string debug_mode)
{
	return debug_mode_set_.find(debug_mode) != debug_mode_set_.end();
}

#endif