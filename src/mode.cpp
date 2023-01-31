#include "mode.h"

#include <set>
#include <string>

std::set<std::string> mode_set_;

void addMode(char *mode) { mode_set_.insert(std::string(mode)); }

bool isMode(std::string mode) {
    return mode_set_.find(mode) != mode_set_.end();
}
