#include <string>

namespace NV {
    namespace Metric {
        namespace Parser {
            inline bool ParseMetricNameString(const std::string& metricName, std::string* reqName, bool* isolated, bool* keepInstances)
            {
                std::string& name = *reqName;
                name = metricName;
                if (name.empty())
                {
                    return false;
                }

                // boost program_options sometimes inserts a \n between the metric name and a '&' at the end
                size_t pos = name.find('\n');
                if (pos != std::string::npos)
                {
                    name.erase(pos, 1);
                }

                // trim whitespace
                while (name.back() == ' ')
                {
                    name.pop_back();
                    if (name.empty())
                    {
                        return false;
                    }
                }

                *keepInstances = false;
                if (name.back() == '+')
                {
                    *keepInstances = true;
                    name.pop_back();
                    if (name.empty())
                    {
                        return false;
                    }
                }

                *isolated = true;
                if (name.back() == '$')
                {
                    name.pop_back();
                    if (name.empty())
                    {
                        return false;
                    }
                }
                else if (name.back() == '&')
                {
                    *isolated = false;
                    name.pop_back();
                    if (name.empty())
                    {
                        return false;
                    }
                }

                return true;
            }
        }
    }
}