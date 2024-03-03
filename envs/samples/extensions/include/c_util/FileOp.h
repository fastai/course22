#include <fstream>
#include <vector>
#include <iostream>

bool WriteBinaryFile(const char* pFileName, const std::vector<uint8_t>& data)
{
    FILE* fp = fopen(pFileName, "wb");
    if (fp)
    {
        if (data.size())
        {
            fwrite(&data[0], 1, data.size(), fp);
        }
        fclose(fp);
    }
    else
    {
        std::cout << "ERROR!! Failed to open " << pFileName << "\n";
        std::cout << "Make sure the file or directory has write access\n";
        return false;
    }
    return true;
}

bool ReadBinaryFile(const char* pFileName, std::vector<uint8_t>& image)
{
    FILE* fp = fopen(pFileName, "rb");
    if (!fp)
    {
        std::cout << "ERROR!! Failed to open " << pFileName << "\n";
        std::cout << "Make sure the file or directory has read access\n";
        return false;
    }

    fseek(fp, 0, SEEK_END);
    const long fileLength = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (!fileLength)
    {
        std::cout << pFileName << " has zero length\n";
        fclose(fp);
        return false;
    }

    image.resize((size_t)fileLength);
    fread(&image[0], 1, image.size(), fp);
    fclose(fp);
    return true;
}