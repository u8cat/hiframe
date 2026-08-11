#pragma once
#include <filesystem>
#include <system_error>

// Executable directory
std::filesystem::path get_executable_directory();

// Create the directory holding `file`, together with any missing parent, unless
// it already exists. Returns the reason of the failure, if any.
std::error_code create_parent_directory(const std::filesystem::path &file);
