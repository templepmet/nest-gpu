#ifndef MODE_H
#define MODE_H

#include <set>
#include <string>

extern std::set<std::string> mode_set_;

void addMode(char *mode);
bool isMode(std::string mode);

#endif
