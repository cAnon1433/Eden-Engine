#pragma once

#include <string>

namespace Eden
{
    // Pure debug label - not read by any system, just something to print
    // in logs/an eventual inspector so "Entity 47" can show up as
    // "Entity 47 (PredatorAgent_03)" instead. Covers both "TagComponent"
    // and "NameComponent" from the TODO - they were the same idea listed
    // as alternate names, so this is the one component, not two.
    struct NameComponent
    {
        std::string name;
    };
}
