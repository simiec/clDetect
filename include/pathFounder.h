#include <iostream>
#include <vector>
#include <filesystem>

namespace visua
{
  namespace fs = std::filesystem;

  class PathFounder{
    public:
      static std::vector<fs::path> getSubDirectories(std::string path){
        std::vector<fs::path> folder_paths;

        // Iterate through the directory
        for (const auto& entry : fs::directory_iterator(path)) {
            // Check if the entry is a directory
            if (fs::is_directory(entry.status())) {
                // Add the directory path to the vector
                folder_paths.push_back(entry.path());
            }
        }

        return folder_paths;
      }

      PathFounder(PathFounder const&) = delete;
      void operator=(PathFounder const&) = delete;

    private:
      PathFounder() {}
  };
} // namespace visua
