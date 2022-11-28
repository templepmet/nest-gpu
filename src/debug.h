#ifndef DEBUG_H
#define DEBUG_H

#include <set>
#include <string>

extern std::set<std::string> debug_mode_set_;

bool isDebugMode(std::string debug_mode);

#endif